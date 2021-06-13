import os
import logging
import json
import numpy as np
import pandas as pd
import pickle
import random
import time

import torch
from torch.utils.data import Dataset

from . import utils
from .skeleton_helper import *
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Skeleton(Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.debug = cfg.SKELETON.DEBUG
        self.random_choose = cfg.SKELETON.RANDOM_CHOOSE
        self.random_move = cfg.SKELETON.RANDOM_MOVE
        self.window_size = cfg.SKELETON.WINDOW_SIZE
        self.mmap = cfg.SKELETON.MMAP

        if split == 'train':
            self.data_path = cfg.SKELETON.TRAIN_DATA_PATH
            self.label_path = cfg.SKELETON.TRAIN_LABEL_FILE
        if split == 'val':
            self.data_path = cfg.SKELETON.TEST_DATA_PATH
            self.label_path = cfg.SKELETON.TEST_LABEL_FILE

        self.load_data(self.mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        data_numpy = utils.pack_pathway_output(self.cfg, data_numpy)

        return data_numpy, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        data_numpy, label = tuple(zip(*batch))

        data_numpy = torch.stack(data_numpy, dim=0)
        label = torch.as_tensor(label)
        return data_numpy, label

# past dataset version
# class Skeleton(torch.utils.data.Dataset):
#     """
#     Dataset based on skeleton
#     """    
#     def __init__(self,cfg, split):
#         self.cfg = cfg
#         self._split = split       
#         self._sample_rate = cfg.DATA.SAMPLING_RATE
#         self._video_length = cfg.DATA.NUM_FRAMES
#         self._seq_len = self._video_length * self._sample_rate
#         self._num_classes = cfg.MODEL.NUM_CLASSES 

#         self._load_data(cfg)

#     def _load_data(self, cfg):
#         """
#         Load frame paths and annotations from files

#         Args:
#             cfg (CfgNode): config
#         """ 
#         if self._split == 'train':
#             self.filedir = cfg.SKELETON.TRAIN_DIR
#             self.label_file = cfg.SKELETON.TRAIN_LABEL_FILE
#         else:
#             self.filedir = cfg.SKELETON.TEST_DIR
#             self.label_file = cfg.SKELETON.TEST_LABEL_FILE
#         self.samples_path = [
#             os.path.join(self.filedir, filename) for filename in os.listdir(self.filedir)
#         ]

#         label_file = self.label_file
#         with open(label_file, 'r+') as f:
#             label_info = f.readlines()
        
#         self.label = np.array(
#             [info.split(',')[2] for info in label_info]
#         )

#         self.C = 3
#         self.V = 25
#         self.M = cfg.SKELETON.PERSON_NUM_OUT

#     def __len__(self):
#         return len(self.samples_path)

#     def __getitem__(self, idx):
#         sample_path = os.path.join(self.filedir, self.samples_path[idx])
#         data_numpy = np.zeros((self.C, self.V, self.num_person_in))
#         json_file_list = [
#             os.path.join(sample_path, sample_name) for sample_name in os.listdir(sample_path)
#             ]
#         for json_file in json_file_list:
#             with open(json_file, 'r') as f:
#                 frame_info = json.load(f)

#             # fill data_numpy
#             # frame_index = frame_info['frame_id']
#             for m, skeleton_info in enumerate(frame_info['skeleton']):
#                 if m >= self.num_person_in:
#                     break
#                 pose = skeleton_info['pose']
#                 score = skeleton_info['score']
#                 data_numpy[0, :, m] = pose[0::2]
#                 data_numpy[1, :, m] = pose[1::2]
#                 data_numpy[2, :, m] = score

#         # centralization
#         data_numpy[0:2] = data_numpy[0:2] - 0.5
#         data_numpy[0][data_numpy[2] == 0] = 0
#         data_numpy[1][data_numpy[2] == 0] = 0

#         # get & check label index
#         label = self.label[idx]


#         return data_numpy, label

