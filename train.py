#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import datetime
import time
import tqdm
import subprocess as sp
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
# Added on V0.3.0
import clearml
import configparser

# Added on V0.3.1
import utils.pytorch_warmup as warmup

#import utils.writer
from models import load_model
from utils.autobatcher import check_train_batch_size
from utils.logger import Logger
from utils.smart_optimizer import smart_optimizer
from utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set, one_cycle, \
    check_amp
from utils.datasets import ListDataset
from utils.augmentations import AUGMENTATION_TRANSFORMS
# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from utils.parse_config import parse_data_config, parse_model_weight_config
from utils.loss import compute_loss, fitness, training_fitness
from test import _evaluate, _create_validation_data_loader
from utils.writer import csv_writer, img_writer_training, img_writer_evaluation, log_file_writer
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
    is_pth_file = False
    list_of_files = os.listdir('checkpoints/')
    full_path = ["checkpoints/{0}".format(x) for x in list_of_files]
    oldest_file = min(full_path, key=os.path.getctime)
    while is_pth_file == False:
        if oldest_file.endswith('.pth'):
            is_pth_file = True
        else:
            full_path.remove(oldest_file)
            oldest_file = min(full_path, key=os.path.getctime)
    os.remove(oldest_file)


def check_folders():
    local_path = os.getcwd()
    # Check if logs folder exists
    logs_path_there = os.path.exists(local_path + "/logs/")
    if not logs_path_there:
        os.mkdir(local_path + "/logs/")
    # Check if checkpoints folder exists
    ckpt_path_there = os.path.exists(local_path + "/checkpoints/")
    if not ckpt_path_there:
        os.mkdir(local_path + "/checkpoints/")
    # Check if checkpoints/best folder exists
    ckpt_best_path_there = os.path.exists(local_path + "/checkpoints/best/")
    if not ckpt_best_path_there:
        os.mkdir(local_path + "/checkpoints/best/")
    # Check if output folder exists
    output_path_there = os.path.exists(local_path + "/output/")
    if not output_path_there:
        os.mkdir(local_path + "/output/")


def run():
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ver = "0.3.14J"
    # Check folders
    check_folders()
    # Create new log file
    f = open("logs/" + date + "_log" + ".txt", "w")
    f.close()
    log_file_writer("Software version: " + ver, "logs/" + date + "_log" + ".txt")
    print_environment_info(ver, "logs/" + date + "_log" + ".txt")

    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--sync_bn", type=int, default=-1,
                        help="Set use of SyncBatchNorm")
    parser.add_argument("--cos_lr", type=int, default=0,
                        help="Set type of scheduler")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="Define which gpu should be used")
    parser.add_argument("--checkpoint_keep_best", type=bool, default=True, help="Should the best checkpoint be saved")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")
    log_file_writer(f"Command line arguments: {args}", "logs/" + date + "_log" + ".txt")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    model_name = data_config["model_name"]
    if model_name == '':
        model_name = str(date)
    else:
        model_name = model_name + '_' + str(date)

    gpu = args.gpu
    auto_eval = True
    best_training_fitness = 9999
    best_fitness = 0.0
    checkpoints_saved = 0
    device = torch.device("cpu")
    epoch_start = ""
    epoch_end = ""
    exec_time = 0
    do_auto_eval = False
    use_smart_optimizer = True
    warmup_run = True
    start_epoch = 0
    #Get model weight eval parameters
    # Create a ConfigParser object
    weight_eval_params = parse_model_weight_config(args.model)
    # Access the parameters from the config file
    w_train = weight_eval_params[0]
    w = weight_eval_params[1]

    ################
    # Create CSV files - version 0.3.8
    ################

    # Create training csv file
    header = ['Iterations', 'Iou Loss', 'Object Loss', 'Class Loss', 'Loss', 'Learning Rate']
    csv_writer(header, args.logdir + "/" + model_name + "_training_plots.csv")

    # Create evaluation csv file
    header = ['Epoch', 'Epochs', 'Precision', 'Recall', 'mAP', 'F1', 'AP CLS', 'Fitness']
    csv_writer(header, args.logdir + "/" + model_name + "_evaluation_plots.csv")

    # Create validation csv file
    header = ['Index', 'Class', 'AP']
    csv_writer(header, f"checkpoints/best/{model_name}_eval_stats.csv")

    ################
    # Create ClearML task - version 0.3.0
    ################
    '''
    This code block checks if the variable clearml_run is true, and if so, reads parameters from a configuration 
    file named clearml.cfg. It then initializes a ClearML task with the specified project and task names, and sets 
    the offline mode if specified in the configuration file. The code also disables auto-connecting to certain 
    frameworks and connects the task to the provided arguments. Finally, it instantiates an OutputModel object 
    for the PyTorch framework with the newly created task.
    '''
    #get clearml parameters
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the config file
    config.read(r'config/clearml.cfg')
    # Access the parameters from the config file
    proj_name = config.get('clearml', 'proj_name')
    #task_name = config.get('clearml', 'task_name')
    offline = config.get('clearml', 'offline')
    if config.get('clearml', 'clearml_save_last') == "True":
        clearml_save_last = True
    else:
        clearml_save_last = False
    if config.get('clearml', 'clearml_run') == "True":
        clearml_run = True
    else:
        clearml_run = False

    if clearml_run:
        task_name = model_name
        if offline == "True":
            # Use the set_offline class method before initializing a Task
            clearml.Task.set_offline(offline_mode=True)
        # Create a new task
        task = clearml.Task.init(project_name=proj_name, task_name=task_name, auto_connect_frameworks={
            'matplotlib': False, 'tensorflow': False, 'tensorboard': False, 'pytorch': True,
            'xgboost': False, 'scikit': True, 'fastai': False, 'lightgbm': False,
            'hydra': False, 'detect_repository': True, 'tfdefines': False, 'joblib': False,
            'megengine': False, 'jsonargparse': True, 'catboost': False})
        # Log model configurations
        task.connect(args)
        # Instantiate an OutputModel with a task object argument
        clearml.OutputModel(task=task, framework="PyTorch")
    '''
    The code first checks if a GPU is available and assigns the device accordingly. 
    If a GPU is available, it assigns the device as "cuda", otherwise it assigns it as "cpu". 
    The device is then printed and logged.
    The code then loads the model using the specified model file and GPU. 
    It also checks if Automatic Mixed Precision (AMP) is enabled, but this feature is not implemented.
    If the code is running with ClearML integration, it logs the model hyperparameters.
    If the verbose flag is set, it prints a summary of the model.
    The code then calculates the batch size based on the model's hyperparameters, height, and AMP. 
    If the calculation fails, it falls back to using the batch size specified in the hyperparameters.
    Finally, the code sets the mini-batch size by dividing the batch size by the number of subdivisions 
    specified in the hyperparameters.
    '''
    # ############
    # GPU memory check and batch setting DONE: Needs more calculations based on parameters -> implemented on 'check_train_batch_size'
    # ############

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DONE:Needs checkup on available gpu memory
    if gpu >= 0:
        if torch.cuda.is_available() is True:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f'Using cuda device - {device}')
    log_file_writer(f'Using cuda device - {device}', "logs/" + date + "_log" + ".txt")

    # ############
    # Create model - Updated on V0.3.14
    # ############

    model = load_model(args.model, gpu, args.pretrained_weights)

    # ############
    # Freeze model layers
    # ############
    # -- Not implemented --

    # ############
    # Check AMP - V.0.3.14
    # ############
    #amp = check_amp(model)  # check AMP -> TODO: causes CUDA overflow error
    amp = False
    # ############
    # Log hyperparameters to clearml
    # ############
    if clearml_run:
        task.connect_configuration(model.hyperparams)
    log_file_writer(f"Model hyperparameters: {model.hyperparams}", "logs/" + date + "_log" + ".txt")

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    batch_size = model.hyperparams['batch']
    # ############
    # Batch size calculation - V0.3.1
    # ############
    try:
        batch_size = check_train_batch_size(model, model.hyperparams['height'], amp)
        sub_div = 1
    except:
        batch_size = model.hyperparams['batch']
        sub_div = model.hyperparams['subdivisions']

    mini_batch_size = batch_size // sub_div
    '''
    The code snippet is creating a dataloader for training and validation data. It first calls the  
    `_create_data_loader`  function to create the training dataloader, passing the path to the training data, 
    mini-batch size, model height, number of CPUs, and a flag indicating whether to use multiscale training. 
    Then, it calls the  `_create_validation_data_loader`  function to create the validation dataloader, 
    passing the path to the validation data, mini-batch size, model height, and number of CPUs.
    After creating the dataloaders, the code snippet creates an optimizer for the model. 
    It checks the optimizer specified in the model's hyperparameters and creates the corresponding optimizer object. 
    The supported optimizers are Adam, SGD, and RMSprop. If an unknown optimizer is specified, a warning message is printed.
    If the optimizer is Adam, the code snippet also creates a learning rate scheduler for warmup. 
    It calculates the total number of steps (number of batches * number of epochs) and passes it to the  
    `CosineAnnealingLR`  scheduler. It also creates a warmup scheduler using the  `UntunedLinearWarmup`  
    class from the  `warmup`  module.
    Finally, the code snippet calculates the number of batches and the number of warmup iterations. 
    The number of batches is the length of the training dataloader, 
    and the number of warmup iterations is set to the maximum of 3 epochs or 100 iterations.
    '''
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

    num_batches = len(dataloader)  # number of batches
    warmup_num = max(
        round(3 * num_batches), 100
    )  # number of warmup iterations, max(3 epochs, 100 iterations)

    # #################
    # Use autoanchor -> Not implemented yet
    # #################

    '''
    if not resume:
            if not args.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)
    '''




    if not use_smart_optimizer:
        # ################
        # Create optimizer
        # ################

        params = [p for p in model.parameters() if p.requires_grad]

        if model.hyperparams['optimizer'] in [None, "adam"]:
            optimizer = optim.AdamW(
                params,
                lr=model.hyperparams['learning_rate'],
                betas=(0.9, 0.999),
                weight_decay=model.hyperparams['decay'],
            )
        elif model.hyperparams['optimizer'] == "sgd":
            optimizer = optim.SGD(
                params,
                lr=model.hyperparams['learning_rate'],
                weight_decay=model.hyperparams['decay'],
                momentum=model.hyperparams['momentum'],
                nesterov=model.hyperparams['nesterov'],
            )
        elif model.hyperparams['optimizer'] == "rmsprop":
            optimizer = optim.RMSprop(params, lr=model.hyperparams['learning_rate'])

        else:
            print("- ⚠ - Unknown optimizer. Please choose between (adam, sgd, rmsprop).")

        if model.hyperparams['optimizer'] == "adam":
            # ################
            # Create lr scheduler for warmup - V 0.3.1 -> works only with adam
            # ################
            num_steps = len(dataloader) * args.epochs
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)


    #DONE: Smart optimizer doesn't seem to work correctly -> Fixed
    else:
        # ################
        # Create smart optimizer - V 0.3.5
        # ################
        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        model.hyperparams['decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        optimizer = smart_optimizer(model, model.hyperparams['optimizer'], float(model.hyperparams['lr0']), float(model.hyperparams['momentum']), float(model.hyperparams['decay']))

        if model.hyperparams['optimizer'] == "adam" or model.hyperparams['optimizer'] == "adamw":
            # ################
            # Create lr scheduler for warmup - V 0.3.1 -> works only with adam
            # ################
            num_steps = len(dataloader) * args.epochs
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    # Scheduler
    if args.cos_lr != -1:
        lf = one_cycle(1, float(model.hyperparams['lrf']), args.epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / args.epochs) * (1.0 - float(model.hyperparams['lrf'])) + float(model.hyperparams['lrf'])  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # #################
    # Use ModelEMA - V0.x.xx -> Not implemented correctly
    # #################

    # EMA
    #ema = ModelEMA(model) if args.ema != -1 else None

    # #################
    # Create GradScaler - V 0.3.14
    # #################
    # Creates a GradScaler once at the beginning of training.
    #scaler = GradScaler()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # #################
    # SyncBatchNorm - V 0.3.14
    # #################
    if args.sync_bn != -1 and torch.cuda.is_available() is True:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        log_file_writer(f'Using SyncBatchNorm()', "logs/" + date + "_log" + ".txt")


    # #################
    # Create Logging variables
    # #################

    # Matplotlib arrays
    iou_loss_array = np.array([])
    obj_loss_array = np.array([])
    cls_loss_array = np.array([])
    batches_array = np.array([])
    loss_array = np.array([])
    lr_array = np.array([])
    epoch_array = np.array([])
    eval_epoch_array = np.array([])
    precision_array = np.array([])
    recall_array = np.array([])
    mAP_array = np.array([])
    f1_array = np.array([])
    # ap_cls_array = np.array([])
    curr_fitness_array = np.array([])
    train_fitness_array = np.array([])
    lr = model.hyperparams['learning_rate']
    scheduler.last_epoch = start_epoch - 1  # do not move
    last_opt_step = -1
    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    print(
        f"- 🎦 - You can monitor training with tensorboard by typing this command into console: tensorboard --logdir {args.logdir} ----")
    print(f"\n- 🔛 - Starting Model {model_name} training... ----")
    '''
    This code snippet is training a model for a certain number of epochs. 
    Inside the training loop, the model is set to training mode using  `model.train()` . 
    Then, for each batch in the dataloader, the gradients are reset using  `optimizer.zero_grad()` .
    Next, the current batch and epoch numbers are calculated to keep track of the progress. 
    The input images and targets are moved to the device and the model is used to generate outputs for the images.
    The loss and its components are computed using the  `compute_loss`  function. 
    There is a conditional block that handles the warmup phase of the training. 
    If the number of integrated batches is less than or equal to the warmup number, the model's optimizer is checked. 
    If it is "adam", the gradients are computed and the optimizer is updated with  `optimizer.step()` . 
    Additionally, a warmup scheduler is used to adjust the learning rate.
    If the optimizer is not "adam", the learning rate is adjusted manually based on the number of batches done so far. 
    The gradients are computed and the optimizer is updated with the adjusted learning rate.
    After the warmup phase, the same conditional block is executed, but this time the learning rate is 
    adjusted using a learning rate scheduler.
    There is an additional conditional block that was added in version 0.3.0. 
    It uses automatic mixed precision (AMP) training to scale the loss and compute scaled gradients. 
    This block is commented out in version 0.3.3B.
    Finally, the learning rate is updated based on the value in the optimizer's parameter groups.
    Overall, this code snippet performs the training loop for a model, handles warmup, 
    and optionally uses AMP for mixed precision training.
    '''

    #Modded on V0.3.14J

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if epoch > 1:
            print(f'- ⏳ - Estimated execution time: {round((exec_time * args.epochs) / 3600, 2)} hours ----')
        if warmup_run:
            print(f'- 🔥 - Running warmup cycle ----')
        model.train()  # Set model to training mode
        mloss = torch.zeros(3, device=device)  # mean losses
        #optimizer.zero_grad()
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        for param in model.parameters():
            param.grad = None
        start_lr = 0.0  # Starting learning rate for warmup
        end_lr = lr  # The final learning rate
        lr = start_lr

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i
            integ_batch_num = batch_i + num_batches * epoch  # number integrated batches (since train start)

            imgs = imgs.to(device, non_blocking=True).float() / 255
            targets = targets.to(device)

            if integ_batch_num <= warmup_num:
                xi = [0, warmup_num]
                # get the progress of warmup
                #progress = integ_batch_num / warmup_num if integ_batch_num <= warmup_num else 1.0
                accumulate = max(1, np.interp(integ_batch_num, xi, [1, num_batches / batch_size]).round())
                if model.hyperparams['optimizer'] in ["adam", "adamw"]:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    optimizer.step()
                    with warmup_scheduler.dampening():
                        scheduler.step()
                else:
                    if batches_done == model.hyperparams['burn_in']:
                        #optimizer.zero_grad()
                        for param in model.parameters():
                            param.grad = None
                    optimizer.step()
                    lr = lr * (batches_done / model.hyperparams['burn_in'])
                    for g in optimizer.param_groups:
                        g['lr'] = float(lr)
            else:
                warmup_run = False
            # Forward
            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)
            if np.isnan(loss.item()) or np.isinf(loss.item()) and args.verbose:
                print("Warning: Loss is NaN or Inf, skipping this update...")
                continue

            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if integ_batch_num - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                #optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                last_opt_step = integ_batch_num
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            #print(f'Batch {batch_i}/{len(dataloader)}, Loss: {loss.item()}, LR: {lr}')
            #############################################################################
            '''
            The code snippet logs the progress of the training process. 
            It prints the loss values and other metrics to the console if the  `verbose`  flag is set to  `True`. 
            It also logs these metrics to TensorBoard for visualization.

            The code snippet also logs the learning rate to TensorBoard and uses 
            the ClearML library to log the loss values and learning rate.
            
            The logged metrics include IoU loss, object loss, class loss, total loss, and batch loss. 
            These metrics provide insights into the performance of the model during training.
            
            The code snippet demonstrates good logging practices by providing informative and 
            organized logs for monitoring and analysis.
            '''
            if loss_components.dim() != 0:
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
                    ("train/loss", float(loss_components[3])),

                ]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)
                #Tensorflow logger - learning rate V0.3.4I
                logger.scalar_summary("train/learning rate", lr, batches_done)

                model.seen += imgs.size(0)

                # ############
                # ClearML progress logger - V0.3.3
                # ############
                if clearml_run:
                    task.logger.report_scalar(title="Train/Losses", series="IoU loss", iteration=batches_done,
                                              value=float(loss_components[0]))
                    task.logger.report_scalar(title="Train/Losses", series="Object loss", iteration=batches_done,
                                              value=float(loss_components[1]))
                    task.logger.report_scalar(title="Train/Losses", series="Class loss", iteration=batches_done,
                                              value=float(loss_components[2]))
                    task.logger.report_scalar(title="Train/Losses", series="Loss", iteration=batches_done,
                                              value=float(loss_components[3]))
                    task.logger.report_scalar(title="Train/Losses", series="Batch loss", iteration=batches_done,
                                              value=to_cpu(loss).item())
                    task.logger.report_scalar(title="Train/Lr", series="Learning rate", iteration=batches_done, value=lr)
            '''
            The code snippet shows the training loop for a YOLOv3 object detection model. 
            It includes the training process, logging of progress, saving of checkpoints, 
            and auto-evaluation of the model's fitness on training metrics. 
            The training progress is logged in a CSV file and image files. 
            Checkpoints are saved every specified number of epochs, and only a limited number of checkpoints are stored. 
            The model's fitness is evaluated using a weighted sum of the IOU loss, class loss, object loss, and 
            total loss. If the auto-evaluated fitness is better than the previous best, 
            it is considered a new best and saved. 
            The ClearML library is used for logging the training fitness.
            '''
            # ############
            # Log training progress writers
            # ############
            #
            # training csv writer
            if loss_components.dim() > 0:
                data = [batches_done,
                        float(loss_components[0]),  # Iou Loss
                        float(loss_components[1]),  # Object Loss
                        float(loss_components[2]),  # Class Loss
                        float(loss_components[3]),  # Loss
                        ("%.17f" % lr).rstrip('0').rstrip('.')  # Learning rate
                        ]
                csv_writer(data, args.logdir + "/" + model_name + "_training_plots.csv")

                # ############
                # ClearML csv reporter logger - V0.3.6
                # ############
                if clearml_run:
                    # Report table - CSV from path
                    csv_url = args.logdir + "/" + model_name + "_training_plots.csv"
                    task.logger.report_table(
                        "Training plots",
                        "training_plots.csv",
                        iteration=batches_done,
                        url=csv_url
                    )

                # img writer
                batches_array = np.concatenate((batches_array, np.array([batches_done])))
                iou_loss_array = np.concatenate((iou_loss_array, np.array([float(loss_components[0])])))
                obj_loss_array = np.concatenate((obj_loss_array, np.array([float(loss_components[1])])))
                cls_loss_array = np.concatenate((cls_loss_array, np.array([float(loss_components[2])])))
                loss_array = np.concatenate((loss_array, np.array([float(loss_components[3].item())])))
                lr_array = np.concatenate((lr_array, np.array([("%.17f" % lr).rstrip('0').rstrip('.')])))
                img_writer_training(iou_loss_array, obj_loss_array, cls_loss_array, loss_array, lr_array, batches_array,
                                    args.logdir + "/" + date)

        # #############
        # Save progress -> changed on version 0.3.11F to save every eval epoch
        # #############
        # Reason to eval epoch change: uploads get stucked when using clearml and larger models
        #
        if epoch % args.evaluation_interval == 0:
            # Save last model to checkpoint file

            # Updated on version 0.3.0 to save only last
            checkpoint_path = f"checkpoints/{model_name}_ckpt_last.pth"
            print(f"- ⏺ - Saving last checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)
            checkpoints_saved += 1

            ############################
            # ClearML last model update - V 0.3.7 -> changed on version 0.3.11F to save every eval epoch
            ############################
            if clearml_run and clearml_save_last:
                task.update_output_model(model_path=f"checkpoints/{model_name}_ckpt_last.pth")

        if auto_eval is True and loss_components.dim() > 0:
            # #############
            # Training fitness evaluation
            # Calculate weighted loss -> smaller losses better training fitness
            # #############
            print("\n- 🔄 - Auto evaluating model on training metrics ----")
            training_evaluation_metrics = [
                float(loss_components[0]),  # Iou Loss
                float(loss_components[1]),  # Object Loss
                float(loss_components[2]),  # Class Loss
                float(loss_components[3]),  # Loss
            ]
            # Updated on version 0.3.12
            #w_train = [0.20, 0.30, 0.30, 0.20]  # weights for [IOU, Class, Object, Loss]
            fi_train = training_fitness(np.array(training_evaluation_metrics).reshape(1, -1), w_train)
            train_fitness = float(fi_train[0])
            if fi_train < best_training_fitness:
                print(f"- ✅ - Auto evaluation result: New best training fitness {fi_train} ----")
                best_training_fitness = fi_train
                do_auto_eval = True
            else:
                print(f"- ❎ - Auto evaluation result: Training fitness {fi_train} ----")

            # ############
            # ClearML training fitness logger - V0.3.4
            # ############
            if clearml_run:
                task.logger.report_scalar(title="Training", series="Fitness", iteration=epoch,
                                          value=float(fi_train[0]))



        '''
        This code snippet is evaluating the performance of a YOLOv3 model on the validation set. 
        It starts by checking if it's time to perform an evaluation based on the specified evaluation interval or 
        if an auto evaluation is triggered. 

        During the evaluation, the model is passed to the  `_evaluate`  function along with the validation dataloader, 
        class names, and other parameters. The function calculates various metrics such as precision, recall, 
        average precision (AP), and F1 score. These metrics are then logged and displayed. 
        Additionally, the best mean average precision (mAP) is updated if the current fitness (weighted combination of 
        precision, recall, mAP, and F1 score) is better than the previous best fitness. 
        
        If the current fitness is better than the previous best fitness, the model's state dictionary is saved as a checkpoint. 
        The evaluation metrics and class-wise APs are also saved in a text file. 
        
        Finally, the evaluation metrics and fitness values are stored in arrays for plotting purposes. 
        These arrays are then saved as CSV and image files for visualization.
        '''
        # ########
        # Evaluate
        # ########
        if epoch % args.evaluation_interval == 0 or do_auto_eval:
            do_auto_eval = False
            # Do evaluation on every epoch for better logging
            print("\n- 🔄 - Evaluating Model ----")
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

                # Log the evaluation metrics
                logger.scalar_summary("validation/precision", float(precision.mean()), epoch)
                logger.scalar_summary("validation/recall", float(recall.mean()), epoch)
                logger.scalar_summary("validation/mAP", float(AP.mean()), epoch)
                logger.scalar_summary("validation/f1", float(f1.mean()), epoch)
                # logger.scalar_summary("validation/ap_class", float(ap_class.mean()), epoch)

                # ############
                # ClearML validation logger - V0.3.3
                # ############
                if clearml_run:
                    task.logger.report_scalar(title="Validation", series="Precision", iteration=epoch,
                                              value=float(precision.mean()))
                    task.logger.report_scalar(title="Validation", series="Recall", iteration=epoch,
                                              value=float(recall.mean()))
                    task.logger.report_scalar(title="Validation", series="mAP", iteration=epoch,
                                              value=float(AP.mean()))
                    task.logger.report_scalar(title="Validation", series="F1", iteration=epoch,
                                              value=float(f1.mean()))
                # ############
                # Current fitness calculation - V0.3.6B
                # ############
                # Updated on version 0.3.12
                w = [0.1, 0.1, 0.6, 0.2, 0.0]  # weights for [P, R, mAP@0.5, f1, ap class]
                fi = fitness(np.array(evaluation_metrics).reshape(1, -1),
                             w)  # weighted combination of [P, R, mAP@0.5, f1]
                curr_fitness = float(fi[0])
                curr_fitness_array = np.concatenate((curr_fitness_array, np.array([curr_fitness])))
                logger.scalar_summary("fitness/model", round(best_fitness, 4), epoch)
                train_fitness_array = np.concatenate((train_fitness_array, np.array([train_fitness])))
                logger.scalar_summary("fitness/training", float(fi_train), epoch)
                print(
                    f"- ➡ - Checkpoint fitness: '{round(curr_fitness, 4)}' (Current best fitness: {round(best_fitness, 4)}) ----")

                if clearml_run:
                    # ############
                    # ClearML fitness logger - V0.3.3
                    # ############
                    if clearml_run:
                        task.logger.report_scalar(title="Checkpoint", series="Fitness", iteration=epoch,
                                                  value=curr_fitness)
                # DONE: This line needs to be fixed -> AssertionError: Tensor should contain one element (0 dimensions). Was given size: 21 and 1 dimensions.
                # img writer - evaluation
                eval_epoch_array = np.concatenate((eval_epoch_array, np.array([epoch])))
                precision_array = np.concatenate((precision_array, np.array([precision.mean()])))
                recall_array = np.concatenate((recall_array, np.array([recall.mean()])))
                mAP_array = np.concatenate((mAP_array, np.array([AP.mean()])))
                f1_array = np.concatenate((f1_array, np.array([f1.mean()])))
                img_writer_evaluation(precision_array, recall_array, mAP_array, f1_array,
                                      curr_fitness_array, train_fitness_array,eval_epoch_array, args.logdir + "/" + date)

                if curr_fitness > best_fitness:
                    best_fitness = curr_fitness
                    checkpoint_path = f"checkpoints/best/{model_name}_ckpt_best.pth"
                    print(f"- ⭐ - Saving best checkpoint to: '{checkpoint_path}'  ----")
                    torch.save(model.state_dict(), checkpoint_path)
                    ############################
                    # ClearML model update - V 3.0.0
                    ############################
                    if clearml_run:
                        task.update_output_model(model_path=f"checkpoints/best/{model_name}_ckpt_best.pth")


                    ############################
                    # Save best checkpoint evaluation stats into csv - V0.3.8
                    #############################

                    precision, recall, AP, f1, ap_class = metrics_output
                    # Gets class AP and mean AP
                    for i, c in enumerate(ap_class):
                        data = [c,  # Class index
                                class_names[c],  # Class name
                                "%.5f" % AP[i],  # Class AP
                                ]

                        csv_writer(data, f"checkpoints/best/{model_name}_eval_stats.csv")

                    #Write mAP value as last line
                    data = ["--",  #
                            'mAP',  #
                            str(round(AP.mean(), 5)),
                            ]
                    csv_writer(data, f"checkpoints/best/{model_name}_eval_stats.csv")

                    # ############
                    # ClearML csv reporter logger - V0.3.8
                    # ############
                    if clearml_run:
                        # Report table - CSV from path
                        csv_url = f"checkpoints/best/{model_name}_eval_stats.csv"
                        task.logger.report_table(
                            "Model evaluation stats",
                            f"{model_name}_eval_stats.csv",
                            iteration=epoch,
                            url=csv_url
                        )

                ############################
                # Model evaluation plots logging
                #############################

                data = [epoch,
                        args.epochs,
                        precision.mean(),  # Precision
                        recall.mean(),  # Recall
                        AP.mean(),  # mAP
                        f1.mean(),  # f1
                        curr_fitness  # Fitness
                        ]
                csv_writer(data, args.logdir + "/" + model_name + "_evaluation_plots.csv")
                # ############
                # ClearML csv reporter logger - V0.3.6
                # ############
                if clearml_run:
                    # Report table - CSV from path
                    csv_url = args.logdir + "/" + model_name + "_evaluation_plots.csv"
                    task.logger.report_table(
                        "Evaluation plots",
                        "evaluation_plots.csv",
                        iteration=epoch,
                        url=csv_url
                    )


                
        epoch_end = time.time()
        exec_time = epoch_end-epoch_start

if __name__ == "__main__":
    run()

# python train.py -m config/yolov3_ITDM_simple.cfg -d config/Nova.data -e 10 -v --pretrained_weights weights/yolov3.weights --checkpoint_interval 1 --evaluation_interval 1
