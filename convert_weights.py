import torch
from models import Darknet

model = Darknet("C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/config/Nova-tiny.cfg")

#model.load_state_dict(torch.load("C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Nova_2023_09_25_08_50_04_ckpt_best.pth", map_location=torch.device('cpu'))) # for loading model on cpu
model.load_state_dict(torch.load("C:/Users/Juha/Documents/AI/Models/Lyra/Lyra-tiny-640/Lyra-tiny_2023_10_06_20_23_43_ckpt_best_6.pth", map_location=torch.device('cpu'))) # for loading model on cpu

model.save_darknet_weights('C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Lyra-tiny_640.weights', cutoff=-1)
print("successfully converted .pth to .weights")