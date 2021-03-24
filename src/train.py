import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel

import shutil
import os
import numpy as np
import time
import argparse
from utils.pytorchtools import EarlyStopping

from dataset.loader import construct_loader
from model.build import build_model
from utils.util import *

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()


def train(rank, world_size, args):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=world_size, rank=rank)
    torch.cuda.set_device(args.local_rank)

    seed = int(time.time() * 256)
    torch.manual_seed(seed)
 
    # ================================================
    # 2) get data and load data
    # ================================================
    train_dataloader = construct_loader(cfg, 'train')
    val_dataloader = construct_loader(cfg, 'val')
    print("""----Data statistics------'
            # feature_dim: [None, %d]
            -------------------------
                """ % 
            (config['features_dim']))

    # ================================================
    # 3) init model/loss/optimizer
    # ================================================

    model = build_model(cfg)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = optim.Adam(model.parameters(),
                        lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    model, optimizer = amp.initialize(model, optimizer)
    # loss_function = F.cross_entropy()

    # ================================================
    # 4) train loop
    # ================================================
    print("|------------------------|")
    print("| train on train dataset |")
    print("|------------------------|")

    early_stopping = EarlyStopping(20, verbose=True)
    writer = SummaryWriter()
    start_time = time.time()
    for epoch in range(args.n_epochs):   
        train_loss_lst = []
        val_loss_lst = []
        train_acc_lst = []
        val_acc_lst = []
        model.train()
        for i, train_dataset in enumerate(train_dataloader):
            train_data, train_label = train_dataset

            if cfg.NUM_GPU:
                train_data.cuda(non_blocking=True)
                train_label.cuda(non_blocking=True)

            optimizer.zero_grad()       #

            # forward + backward + optimize
            train_outputs = model(train_data)
            train_loss = F.cross_entropy(train_outputs, train_label.long())

            adjust_lr(optimizer, epoch, config['learning_rate'])

            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # train_loss.backward()     
            optimizer.step()

            train_acc = accuracy(train_outputs, train_label.long())

            train_acc_lst.append(train_acc)
            train_loss_lst.append(train_loss)
            # print(
            # "Train Phase, Epoch:{} [{}/{}], Train_Loss:{}, Train_Accuracy:{}"
            # .format(
            #     epoch, i, int(len(train_dataloader.dataset)/len(train_data)), 
            #     train_loss.item(), train_acc
            #         )
            #     )
        train_avg_loss = sum(train_loss_lst)/i
        train_avg_acc = sum(train_acc_lst)/i
    # ================================================
    # 5) evaluate on validation dataset
    # ================================================

        model.eval()
        for v ,val_dataset in enumerate(val_dataloader):
            val_data, val_label = val_dataset

            # val_data = val_data.long()
            # val_label = val_label.long()

            val_outputs = model(val_data)
            val_loss = F.cross_entropy(val_outputs, val_label.long())
            val_acc = accuracy(val_outputs, val_label)

            val_acc_lst.append(val_acc)
            val_loss_lst.append(val_loss)

        val_avg_acc = sum(val_acc_lst)/v
        val_avg_loss = sum(val_loss_lst)/v
        print("Train Phase, Epoch:{}, Train_avg_loss:{}, Val_avg_loss:{},Train_avg_acc:{}, Val_avg_acc:{}"
        .format(epoch, train_avg_loss, val_avg_loss, train_avg_acc, val_avg_acc))
        early_stopping(val_avg_loss, model)
        if early_stopping.early_stop:
            print('|------- Early Stop ------|')
            end_time = time.time()
            print("Total spend time:{}s".format(end_time-start_time))
            break

        writer.add_scalar('Loss', train_avg_loss, epoch)
        writer.add_scalar('Accuracy', train_avg_acc, epoch)

    # ================================================
    # 6) save model
    # ================================================
    torch.save(model, 'checkpoints/LeNet_model.pth')



if __name__ == '__main__':
    mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))