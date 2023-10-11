
#import paramiko
import time
import os
import time
import glob
import os
import datetime

import cv2
import numpy as np
import torch
from PIL import Image

from detect import detect_image, detect_images
from models import load_model
from utils.utils import load_classes, rescale_boxes


#from utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info

def get_files_in_dir(directory):
    # Get a list of all daily sub-directories
    daily_dirs = glob.glob(directory + "/*/")

    # Sort the directories by name in decreasing order and pick the first one
    latest_daily_dir = sorted(daily_dirs, reverse=True)[0]

    files = set()
    for dir_path, dir_names, file_names in os.walk(latest_daily_dir):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            files.add(file_path)
    return files

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def _write_json(image_path, detections, img_size, output_path, classes):
    img = np.array(Image.open(image_path))
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    for x1, y1, x2, y2, conf, cls_pred in detections:
        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

    '''
    import json
    import time
    import os
    import random
    
    def get_image_paths_and_detections(folder):
        image_files = [f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')]
        data = []
        for image_file in image_files:
            num_det = random.randint(1, 5)  # for example, generate random number of detections 
            detections = [{'detection': f'det_{i+1}', 'confidence': round(random.uniform(0, 1), 2)} for i in range(num_det)]
            data.append({
            'image_path': os.path.join(folder, image_file),
            'detections': detections
            })
        return data
    
    folder_path = 'path_to_your_folder_of_images'  # replace with your actual folder path
    
    # Get image paths and random detections
    image_paths_and_detections = get_image_paths_and_detections(folder_path)
    
    # Create data for JSON
    data_for_json = {
        "timestamp": time.time(),
        "folder_path": folder_path,
        "images": image_paths_and_detections
    }
    
    # Convert the data to JSON
    json_data = json.dumps(data_for_json, indent=4)
    
    # Print the JSON data
    print(json_data)
    '''

def monitor_local_folder(directory, interval,classes, model_path,gpu, weights_path,img_size,conf_thres,nms_thres):
    #Load model and needed config files
    print('Loading model...')
    model = load_model(model_path, gpu, weights_path)
    print('Model loaded...')
    print(f"Start monitoring folder: {directory}")

    files_before = get_files_in_dir(directory)
    for file in files_before:
        print(f"File detected: {file}")

    while True:
        time.sleep(interval)

        files_after = get_files_in_dir(directory)

        added_files = files_after - files_before
        if added_files:
            img_paths = []
            for file in added_files:
                print(f"New file detected: {file}")
                img_paths.append(file)
                img = cv2.imread(file)
                img, orig_img, dim = prep_image(img,img_size)
                print(f"Detecting objects in new images...")
                detections, imgs = detect_image(model, img, img_size, conf_thres, nms_thres)
                _write_json(file,detections[0],img_size,"",classes)

            files_before = files_after
            print('Detections for new files done...')


def monitor_folder_ssh(host, port, username, password, directory, interval):
    pass
    '''
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port, username, password)

    try:
        sftp = client.open_sftp()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        client.close()
        return

    print(f"Start monitoring folder: {directory}")

    files_before = get_files_in_dir(sftp, directory)

    while True:
        time.sleep(interval)

        try:
            files_after = get_files_in_dir(sftp, directory)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            break

        added_files = files_after - files_before
        if added_files:
            for file in added_files:
                print(f"New file detected: {file}")
            files_before = files_after
    sftp.close()
    client.close()
    '''
def run():
    date = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    ver = "0.1.0"
    #print_environment_info(ver, "output/" + date + "_detect" + ".txt")
    # Parse config file
    directory = "C:/Users/Juha/Documents/AI/datasets/aug-2023.tar/aug-2023/srv/data_fetching/road_camera_data/data/datalake/digitraffic/images/"
    json_path = ""
    # Connection parameters
    host = "hostname"
    port = 22  # default SSH port
    username = "username"
    password = "password"
    classes_path = "config/Lyra.names"
    #directory = "/path/to/your/folder"
    #print(f"Command line arguments: {args}")
    #Create_new detect_file
    classes = load_classes(classes_path)
    conf_thres = 0.35
    nms_thres = 0.5
    img_size = 640
    model = "config/Lyra-tiny.cfg"
    weights = "weights/Lyra-tiny_640.weights"
    gpu = -1
    # List of class names
    monitor_local_folder(directory, 10,classes,model,gpu,weights,img_size,conf_thres,nms_thres)


if __name__ == '__main__':
    run()