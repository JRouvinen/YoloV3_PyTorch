#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import datetime

import tqdm
import subprocess as sp
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import utils.writer
from models import load_model
from utils.logger import Logger
from utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from utils.datasets import ListDataset
from utils.augmentations import AUGMENTATION_TRANSFORMS
# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from utils.parse_config import parse_data_config
from utils.loss import compute_loss, fitness
from test import _evaluate, _create_validation_data_loader
from utils.writer import csv_writer, img_writer_training, img_writer_evaluation
from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def find_and_del_last_ckpt():
    list_of_files = os.listdir('checkpoints/')
    full_path = ["checkpoints/{0}".format(x) for x in list_of_files]
    oldest_file = min(full_path, key=os.path.getctime)
    os.remove(oldest_file)

def run():
    ver = "0.1.0"
    print_environment_info(ver)
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="Define which gpu should be used")
    parser.add_argument("--checkpoint_store", type=int, default=5, help="How many checkpoints should be stored")
    parser.add_argument("--checkpoint_keep_best", type=bool, default=True, help="How many checkpoints should be stored")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Create training csv file
    header = ['Epoch', 'Epochs','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']
    date = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    csv_writer(header,args.logdir+"/"+date+"_training_plots.csv")

    # Create training csv file
    header = ['Epoch', 'Epochs', 'Precision', 'Recall', 'mAP', 'F1', 'AP CLS']
    date = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    csv_writer(header, args.logdir + "/" + date + "_evaluation_plots.csv")

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    gpu = args.gpu
    checkpoints_to_keep = args.checkpoint_store
    best_fitness = 0.0
    checkpoints_saved = 0
    # ############
    # GPU memory check and setting
    # ############
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DONE:Needs checkup on available gpu memory
    if gpu >= 0:
        if torch.cuda.is_available() is True:
            device = torch.device("cuda")
        # clear GPU cache
        if device == torch.cuda.FloatTensor:
            torch.cuda.empty_cache()
            available_gpu_mem = get_gpu_memory()
            if available_gpu_mem[0] < 5000:
                print(f'Not enough free GPU memory available [min 6GB] -> switching into cpu')
                device = torch.device("cpu")

                gpu = -1
    else:
        device = torch.device("cpu")
    print(f'Using cuda device - {device}')

    # ############
    # Create model
    # ############

    model = load_model(args.model, gpu,args.pretrained_weights)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if model.hyperparams['optimizer'] in [None, "adam"]:
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif model.hyperparams['optimizer'] == "sgd":
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # #################
    # Create Logging variables
    # #################

    # Matplotlib arrays
    iou_loss_array = np.array([])
    obj_loss_array = np.array([])
    cls_loss_array = np.array([])
    h_loss_array = np.array([])
    loss_array = np.array([])
    lr_array = np.array([])
    epoch_array = np.array([])
    eval_epoch_array = np.array([])
    precision_array = np.array([])
    recall_array = np.array([])
    mAP_array = np.array([])
    f1_array = np.array([])
    ap_cls_array = np.array([])
    curr_fitness_array = np.array([])
    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    print(f"You can monitor training with tensorboard by typing this command into console: tensorboard --logdir {args.logdir}")

    for epoch in range(1, args.epochs + 1):

        print("\n---- Training Model ----")
        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # ############
        # Log progress writers
        # ############
        #
        # training csv writer
        data = [epoch,
                args.epochs,
                float(loss_components[0]), #Iou Loss
                float(loss_components[1]), #Object Loss
                float(loss_components[2]), #Class Loss
                float(loss_components[3]), #Loss
                ("%.17f" % lr).rstrip('0').rstrip('.')
                ]
        csv_writer(data, args.logdir + "/" + date + "_training_plots.csv")

        # img writer
        epoch_array = np.concatenate((epoch_array, np.array([epoch])))
        iou_loss_array = np.concatenate((iou_loss_array, np.array([float(loss_components[0])])))
        obj_loss_array = np.concatenate((obj_loss_array, np.array([float(loss_components[1])])))
        cls_loss_array = np.concatenate((cls_loss_array, np.array([float(loss_components[2])])))
        loss_array = np.concatenate((loss_array, np.array([float(loss_components[3])])))
        lr_array = np.concatenate((lr_array, np.array([("%.17f" % lr).rstrip('0').rstrip('.')])))
        img_writer_training(iou_loss_array, obj_loss_array, cls_loss_array, loss_array, lr_array, epoch_array, args.logdir + "/" + date)
        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        # DONE: needs worker to count how many checkpoints should be stored or best stored
        if epoch % args.checkpoint_interval == 0:
            if checkpoints_saved == checkpoints_to_keep:
                find_and_del_last_ckpt()
                checkpoints_saved -= 1
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)
            checkpoints_saved += 1


        # ########
        # Evaluate
        # ########
        # Update best mAP
        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose,
                device=device
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    precision.mean(),
                    recall.mean(),
                    AP.mean(),
                    f1.mean(),
                    ap_class.mean()
                ]
                #print("Metrics output: ", metrics_output)
                #print("Evaluation metrics: ", evaluation_metrics)
                # Log the evaluation metrics
                logger.scalar_summary("validation/precision", float(precision.mean()), epoch)
                logger.scalar_summary("validation/recall", float(recall.mean()), epoch)
                logger.scalar_summary("validation/mAP", float(AP.mean()), epoch)
                logger.scalar_summary("validation/f1", float(f1.mean()), epoch)
                logger.scalar_summary("validation/ap_class", float(ap_class.mean()), epoch)
                #DONE: This line needs to be fixed -> AssertionError: Tensor should contain one element (0 dimensions). Was given size: 21 and 1 dimensions.
                #img writer - evaluation
                eval_epoch_array = np.concatenate((eval_epoch_array, np.array([epoch])))
                precision_array = np.concatenate((precision_array, np.array([precision.mean()])))
                recall_array = np.concatenate((recall_array, np.array([recall.mean()])))
                mAP_array = np.concatenate((mAP_array, np.array([AP.mean()])))
                f1_array = np.concatenate((f1_array, np.array([f1.mean()])))
                ap_cls_array = np.concatenate((ap_cls_array, np.array([ap_class.mean()])))
                #img_writer_evaluation(precision_array, recall_array, mAP_array, f1_array, ap_cls_array,curr_fitness,eval_epoch_array, args.logdir + "/" + date)
                #evaluate csv writer
                data = [epoch,
                        args.epochs,
                        precision.mean(),  # Precision
                        recall.mean(),  # Recall
                        AP.mean(),  # mAP
                        f1.mean(),  # f1
                        ap_class.mean() # AP
                        ]
                csv_writer(data, args.logdir + "/" + date + "_evaluation_plots.csv")
            if metrics_output is not None:
                fi = fitness(np.array(evaluation_metrics).reshape(1, -1))  # weighted combination of [P, R, mAP@0.5, f1]
                curr_fitness = float(fi[0])
                curr_fitness_array = np.concatenate((curr_fitness_array, np.array([curr_fitness])))
                print(f"---- Checkpoint fitness: '{round(curr_fitness, 4)}' ----")
                #print("Best fitness: ", best_fitness)
                if curr_fitness > best_fitness:
                    best_fitness = curr_fitness
                    checkpoint_path = "checkpoints/yolov3_ckpt_best.pth"
                    print(f"---- Saving best checkpoint to: '{checkpoint_path}' ----")
                    torch.save(model.state_dict(), checkpoint_path)
                img_writer_evaluation(precision_array, recall_array, mAP_array, f1_array, ap_cls_array,
                                      curr_fitness_array, eval_epoch_array, args.logdir + "/" + date)


if __name__ == "__main__":
    run()

# python train.py -m config/yolov3_ITDM_simple.cfg -d config/ITDM_simple.data -e 10 -v --pretrained_weights weights/yolov3.weights --checkpoint_interval 1 --evaluation_interval 1