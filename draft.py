import torch
import torch.nn as nn
import torch.nn
from torch.nn import functional as F
import numpy as np
import pickle

label_path = 'data/ucf101_skeleton/train_label.pkl'
with open(label_path, 'rb') as f:
    label_info = pickle.load(f)
label = label_info[1]
label_idx = list(set(label)) 
print(label_idx)

