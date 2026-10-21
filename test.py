import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
from utils import label2onehot, Quantization_S, calc_map_k,calc_simY,eval_turn




device = torch.device("cuda:0")
model_path = './model/model_CUB_allbit.pth'
model = torch.load(model_path, map_location=device)



test_map_64, test_map_48, test_map_32, test_map_16, test_acc = eval_turn(model,
                                                                         dataloader['val'], train_labels_onehot,
                                                                         device, train_codes,
                                                                         model_quan_S.Codebook.cpu())