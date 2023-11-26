import torch
from models import Darknet
from utils.parse_config import parse_hyp_config

hyp_config = parse_hyp_config("C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/config/hyp.cfg")
model = Darknet("C:/Users/Juha/Documents/AI/Models/Orion/Orion-tiny_v4/Orion-tiny_v2.cfg",hyp_config)

#model.load_state_dict(torch.load("C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Nova_2023_09_25_08_50_04_ckpt_best.pth", map_location=torch.device('cpu'))) # for loading model on cpu
model.load_state_dict(torch.load("C:/Users/Juha/Documents/AI/Models/Orion/experiments/exp8/Orion-tiny_832_v2_2023_11_15_09_27_06_ckpt_best.pth", map_location=torch.device('cpu'))) # for loading model on cpu

model.save_darknet_weights('C:/Users/Juha/PycharmProjects/YoloV3_PyTorch/weights/Orion-tiny_832_v2_exp8.weights', cutoff=-1)
print("successfully converted .pth to .weights")