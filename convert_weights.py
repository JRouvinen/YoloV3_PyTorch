import torch
from models import Darknet
import os

def check_file_exists(file_path):
    if os.path.exists(file_path):
        print("File exists")
    else:
        print("File does not exist")
        exit()

#Get model config and weights
model_cfg = input('Enter model config path: ')
model_cfg = model_cfg.strip()
model_cfg = model_cfg.replace("\\", "/")
#Check if files exists
check_file_exists(model_cfg)
model_path = input('Enter model weights (*.pth) path: ')
model_path = model_path.strip()
model_path = model_path.replace("\\", "/")
#Check if files exists
check_file_exists(model_path)

#Load model
model = Darknet(model_cfg)

#Convert .pth to .weights
#model.load_state_dict(torch.load("C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Nova_2023_09_25_08_50_04_ckpt_best.pth", map_location=torch.device('cpu'))) # for loading model on cpu
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # for loading model on cpu

#Parse model name
model_name = model_cfg.split('/')[-1].split('.')[0]

#Save model
model.save_darknet_weights('C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Lyra-tiny_800.weights', cutoff=-1)
print("successfully converted .pth to .weights")