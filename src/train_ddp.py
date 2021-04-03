from sys import path
import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel

import shutil
import logging
import os
import numpy as np
import time
import argparse
import random

from utils.pytorchtools import EarlyStopping
from dataset.loader import construct_loader
from model.build import build_model
from utils.util import *
from configs.deafualt import get_cfg


parser = argparse.ArgumentParser()
parser.add_argument(
        '--local_rank',
        default=-1, 
        type=int,
        help='node rank for distributed training'
    )
parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/ImgDataset_configs.yaml",
        type=str,
    )
parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:7999",
        type=str,
    )
parser.add_argument(
        "--world_size",
        help="Number of world_size",
        default=4,
        type=int,
    )                       
args = parser.parse_args()


def main_work(world_size, rank, args, cfg):
    # dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=world_size, rank=rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(world_size)

    seed = int(time.time() * 256)
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.deterministic = cfg.CUDNN_DETERMINISTIC

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=20, format='%(asctime)s - %(message)s')
 
  
    # load data
    train_dataloader = construct_loader(cfg, 'train')
    val_dataloader = construct_loader(cfg, 'val')

    # instantiate model
    model = build_model(cfg)
    model.cuda()
    optimizer = optim.Adam(model.parameters(),
                        lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
    model, optimizer = amp.initialize(model, optimizer)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.world_size)

    cudnn.benchmark = True

    loss_function = F.cross_entropy().cuda()

    print("|------------------------|")
    print("| train on train dataset |")
    print("|------------------------|")
    train(train_dataloader, model, loss_function, optimizer, args, cfg)
    if cfg.TRAIN.VALIDATION:
        validation(val_dataloader, model)
    logging.FileHandler('logs/{}_log.txt'.format(time.strftime(r"%Y-%m-%d-%H_%M_%S", time.localtime())))

def train(dataloader, model, loss_func, optimizer, args, cfg):
    early_stopping = EarlyStopping(20, verbose=True, path='checkpoints/model.pth', trace_func=logging.info)
    writer = SummaryWriter()
    start_time = time.time()
    for epoch in range(args.n_epochs):   
        train_loss_lst = []
        
        train_acc_lst = []
        
        model.train()
        for i, train_dataset in enumerate(dataloader):
            train_data, train_label = train_dataset

            if cfg.NUM_GPU:
                train_data.cuda(non_blocking=True)
                train_label.cuda(non_blocking=True)
                torch.distributed.barrier()
            

            optimizer.zero_grad()       #

            # forward + backward + optimize
            train_outputs = model(train_data)
            train_loss = loss_func(train_outputs, train_label.long())

            adjust_lr(optimizer, epoch, cfg.SOLVER.BASE_LR)

            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # train_loss.backward()     
            optimizer.step()

            train_acc = accuracy(train_outputs, train_label.long())

            train_acc_lst.append(train_acc)
            train_loss_lst.append(train_loss)
            
        train_avg_loss = sum(train_loss_lst)/i
        train_avg_acc = sum(train_acc_lst)/i
        logging.info("Train Phase, Epoch:{}/{}, Train_avg_loss:{}, Train_avg_acc:{}"
        .format(epoch, cfg.SOLVER.MAX_EPOCH, train_avg_loss, train_avg_acc))
        early_stopping(train_avg_loss, model)
        if early_stopping.early_stop:
            print('|------- Early Stop ------|')
            end_time = time.time()
            logging.info("Total spend time:{}s".format(end_time-start_time))
            break

        writer.add_scalar('Loss', train_avg_loss, epoch)

        writer.add_scalar('Accuracy', train_avg_acc, epoch)
    

def validation(dataloader, model):
    val_loss_lst = []
    val_acc_lst = []
    model.eval()
    for v ,val_dataset in enumerate(dataloader):
        val_data, val_label = val_dataset


        val_outputs = model(val_data)
        val_loss = F.cross_entropy(val_outputs, val_label.long())
        val_acc = accuracy(val_outputs, val_label)

        val_acc_lst.append(val_acc)
        val_loss_lst.append(val_loss)

    val_avg_acc = sum(val_acc_lst)/v
    val_avg_loss = sum(val_loss_lst)/v
    logging.info("Val Phase, Val_avg_loss:{}, Val_avg_acc:{}".format(val_avg_loss, val_avg_acc))
    
    



if __name__ == '__main__':
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    mp.spawn(main_work, nprocs=args.world_size, args=(args.local_rank, args, cfg))
    # train(args.world_size, cfg, args)