import csv
import matplotlib.pyplot as plt

def csv_writer(data, filename):
    #header = ['Epoch', 'Epochs','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']
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
    fig = plt.figure(layout="constrained")
    fig.set_dpi(1024)
    ax_array = fig.subplots(2, 3, squeeze=False)
    # Using Numpy to create an array x
    x = epoch

    # Plot for iou loss
    ax_array[0, 0].set_ylabel('IoU loss')
    ax_array[0, 0].plot(x, iou_loss, marker = 'o')
    #ax_array[0, 0].invert_yaxis()
    ax_array[0, 0].set_xlabel('Epoch')

    # Plot for obj loss
    ax_array[0, 1].set_ylabel('Object loss')
    ax_array[0, 1].plot(x, obj_loss, marker = 'o')
    ax_array[0, 1].set_xlabel('Epoch')

    # Plot for cls loss
    ax_array[0, 2].set_ylabel('Class loss')
    ax_array[0, 2].plot(x, cls_loss, marker = 'o')
    ax_array[0, 2].set_xlabel('Epoch')

    # Plot for loss
    ax_array[1, 0].set_ylabel('Loss')
    ax_array[1, 0].plot(x, loss, marker = 'o')
    ax_array[1, 0].set_xlabel('Epoch')

    # Plot for learning rate
    ax_array[1, 1].set_ylabel('Learning rate')
    ax_array[1, 1].plot(x, lr, marker = 'o')
    ax_array[1, 1].set_xlabel('Epoch')


    fig.savefig(filename+'_training_metrics.png')
    plt.close()

def img_writer_evaluation(precision, recall, mAP, f1, epoch, filename):
    # img_writer_data = global_step,x_loss,y_loss,w_loss,h_loss,conf_loss,cls_loss,loss,recall,precision
    # Placing the plots in the plane
    fig = plt.figure(layout="constrained")
    fig.set_dpi(1024)
    ax_array = fig.subplots(2, 2, squeeze=False)
    # Using Numpy to create an array x
    x = epoch

    # Plot for precision
    ax_array[0, 0].set_ylabel('Precision')
    ax_array[0, 0].plot(x, precision, marker = 'o')
    #ax_array[0, 0].invert_yaxis()
    ax_array[0, 0].set_xlabel('Epoch')

    # Plot for recall
    ax_array[0, 1].set_ylabel('Recall')
    ax_array[0, 1].plot(x, recall, marker = 'o')
    ax_array[0, 1].set_xlabel('Epoch')

    # Plot for mAP
    ax_array[1, 0].set_ylabel('mAP')
    ax_array[1, 0].plot(x, mAP, marker = 'o')
    ax_array[1, 0].set_xlabel('Epoch')

    # Plot for f1
    ax_array[1, 1].set_ylabel('F1')
    ax_array[1, 1].plot(x, f1, marker = 'o')
    ax_array[1, 1].set_xlabel('Epoch')


    fig.savefig(filename+'_evaluation_metrics.png')
    plt.close()