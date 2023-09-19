import csv
import matplotlib.pyplot as plt
import numpy as np
def csv_writer(data, filename):
    #header = ['Iterations','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']
    #header = ['Epoch', 'Epochs', 'Precision', 'Recall', 'mAP', 'F1']
    #log_path = filename.replace("checkpoints", "")
    with open(filename, 'a', encoding='UTF8') as f:
        table_writer = csv.writer(f)
        # write the data
        table_writer.writerow(data)
    f.close()

def img_writer_training(iou_loss, obj_loss, cls_loss, loss, lr, epoch, filename):
    #header = ['Epoch', 'Epochs','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']    # img_writer_data = global_step,x_loss,y_loss,w_loss,h_loss,conf_loss,cls_loss,loss,recall,precision
    #log_path = filename.replace("checkpoints", "")
    # Placing the plots in the plane
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    #fig.set_dpi(1240)
    ax_array = fig.subplots(2, 3, squeeze=False)
    # Using Numpy to create an array x
    x = epoch

    # Plot for iou loss
    ax_array[0, 0].set_ylabel('IoU loss')
    ax_array[0, 0].plot(x, iou_loss, marker = 'o')
    ax_array[0, 0].grid(axis='y', linestyle='-')
    ax_array[0, 0].set_xlabel('Iteration')

    # Plot for obj loss
    ax_array[0, 1].set_ylabel('Object loss')
    ax_array[0, 1].plot(x, obj_loss, marker = 'o')
    ax_array[0, 1].grid(axis='y', linestyle='-')
    ax_array[0, 1].set_xlabel('Iteration')

    # Plot for cls loss
    ax_array[0, 2].set_ylabel('Class loss')
    ax_array[0, 2].plot(x, cls_loss, marker = 'o')
    ax_array[0, 2].grid(axis='y', linestyle='-')
    ax_array[0, 2].set_xlabel('Iteration')

    # Plot for loss
    ax_array[1, 0].set_ylabel('Loss')
    ax_array[1, 0].plot(x, loss, marker = 'o')
    ax_array[1, 0].grid(axis='y', linestyle='-')
    ax_array[1, 0].set_xlabel('Iteration')

    # Plot for learning rate
    ax_array[1, 1].set_ylabel('Learning rate')
    ax_array[1, 1].plot(x, lr, marker = 'o')
    # https://stackoverflow.com/questions/21393802/how-to-specify-values-on-y-axis-of-a-matplotlib-plot
    ax_array[1, 1].grid(axis='y', linestyle='-')
    ax_array[1, 1].get_autoscaley_on()
    ax_array[1, 1].invert_yaxis()
    if x > 5:
        ax_array[1, 1].set_yscale('log')
    ax_array[1, 1].set_xlabel('Iteration')


    fig.savefig(filename+'_training_metrics.png')
    # displaying the title
    plt.title(filename)
    plt.close()

def img_writer_evaluation(precision, recall, mAP, f1, ckpt_fitness,epoch, filename):
    #img_writer_evaluation(precision_array, recall_array, mAP_array, f1_array, ap_cls_array, curr_fitness_array, eval_epoch_array, args.logdir + "/" + date)
    # Placing the plots in the plane
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    #fig.set_dpi(1240)
    ax_array = fig.subplots(2, 3, squeeze=False)
    # Using Numpy to create an array x
    x = epoch

    # Plot for precision
    ax_array[0, 0].set_ylabel('Precision')
    ax_array[0, 0].plot(x, precision, marker = 'o')
    ax_array[0, 0].grid(axis='y', linestyle='-')
    #ax_array[0, 0].invert_yaxis()
    ax_array[0, 0].set_xlabel('Epoch')
    #ax_array[0, 0].set_ybound([0, 1])


    # Plot for recall
    ax_array[0, 1].set_ylabel('Recall')
    ax_array[0, 1].plot(x, recall, marker = 'o')
    ax_array[0, 1].grid(axis='y', linestyle='-')
    ax_array[0, 1].set_xlabel('Epoch')
    #ax_array[0, 1].set_ybound([0, 1])


    # Plot for mAP
    ax_array[0, 2].set_ylabel('mAP')
    ax_array[0, 2].plot(x, mAP, marker = 'o')
    ax_array[0, 2].grid(axis='y', linestyle='-')
    ax_array[0, 2].set_xlabel('Epoch')
    #ax_array[0, 2].set_ybound([0, 1])


    # Plot for f1
    ax_array[1, 0].set_ylabel('F1')
    ax_array[1, 0].plot(x, f1, marker = 'o')
    ax_array[1, 0].grid(axis='y', linestyle='-')
    ax_array[1, 0].set_xlabel('Epoch')
    #ax_array[1, 0].set_ybound([0, 1])

    ''' 
    #Dropped on version 0.3.1
    # Plot for ap_cls
    ax_array[1, 1].set_ylabel('AP CLS')
    ax_array[1, 1].plot(x, ap_cls, marker='o')
    ax_array[1, 1].grid(axis='y', linestyle='-')
    ax_array[1, 1].set_xlabel('Epoch')
    #ax_array[1, 1].set_ybound([-1, ])
    '''
    # Plot for ckpt fitness
    ax_array[1, 2].set_ylabel('CKPT FITNESS')
    ax_array[1, 2].plot(x, ckpt_fitness, marker='o')
    ax_array[1, 2].grid(axis='y', linestyle='-')
    ax_array[1, 2].set_xlabel('Epoch')
    #ax_array[1, 2].set_ybound([0, 10])

    fig.savefig(filename+'_evaluation_metrics.png')
    plt.close()

def log_file_writer(data, filename):
    #log_path = filename.replace("checkpoints", "")
    with open(filename, 'a', encoding='UTF8') as f:
        # write the data
        f.write("\n"+data)
    f.close()
