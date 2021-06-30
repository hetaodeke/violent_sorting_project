import argparse
import os
import random
import shutil
import time
import warnings
import logging

import torch
from torch.multiprocessing.spawn import spawn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter


from dataset.loader import construct_loader
from model.build import build_model
from utils.util import *
from utils.meter import AverageMeter, ProgressMeter
from configs.deafualt import get_cfg


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--nprocs',
                    default=2,
                    type=int,
                    metavar='N',
                    help='number of data loading workers')
parser.add_argument('--num_gpus', default=2, type=int, help='number of gpus')
parser.add_argument('--gpu_idx', default='1,2', type=str, help='visible devices number')
parser.add_argument("--cfg",
                    dest="cfg_file",
                    help="Path to the config file",
                    default="configs/ImgDataset_configs.yaml",
                    type=str)
parser.add_argument('--log_path',
                    default='logs/{}.log'.format(time.strftime(r"%Y-%m-%d-%H_%M_%S", time.localtime())),
                    help='path of log file')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', 
                    default=int(time.time() * 256), 
                    type=int, 
                    help='seed for initializing training. ')
args = parser.parse_args()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main_worker(rank, nprocs, args ,cfg):
    best_acc1 = .0
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345', world_size=args.nprocs, rank=rank)


    if is_master_proc(args) and args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training.' 
                      'This will turn on the CUDNN deterministic setting,' 
                      'which can slow down your training considerably!' 
                      'You may see unexpected behavior when restarting' 
                      'from checkpoints.')

    # create model
    model = build_model(cfg)
    model.cuda(rank)


    # define loss function (criterion) and optimizer
    # criterion = nn.BCELoss().cuda(rank)
    criterion = nn.CrossEntropyLoss().cuda(rank)

    if cfg.TRAIN.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # model, optimizer = amp.initialize(model, optimizer)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    cudnn.benchmark = True

    # Data loading code
    train_dataloader = construct_loader(cfg, 'train')
    val_dataloader = construct_loader(cfg, 'val')

    # get logget 
    logger = get_logger(args)
    logger.info('Parameter:{ Batch_size:' + str(cfg.TRAIN.BATCH_SIZE) + 
                '\tbase_lr:' + str(cfg.SOLVER.BASE_LR) + '\tlr_decay:' + str(cfg.SOLVER.LR_DECAY) + 
                '\tModel:' + cfg.MODEL.MODEL_NAME + '}')
    # tesorboard writer
    writer = SummaryWriter()

    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        # train for one epoch
        loss = train(train_dataloader, model, criterion, optimizer, epoch, rank, args, logger, cfg)

        # evaluate on validation set
        acc1, acc5 = validate(val_dataloader, model, criterion, rank, args, logger, cfg)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)

        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Acc@1', acc1, epoch)
        writer.add_scalar('Acc@5', acc5, epoch)
        writer.flush()
        writer.close()
    return print("Train finished! to see train infomation in log/, to see train result in runs/ by using tensorboard command")

def train(train_loader, model, criterion, optimizer, epoch, rank, args, logger, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    for iter, traindata in enumerate(train_loader):
        data, label = traindata
        if isinstance(data, (list,)):
            for i in range(len(data)):
                data[i] = data[i].cuda(rank)
        else:
            data = data.cuda(rank)
        # label = torch.nn.functional.one_hot(label, cfg.MODEL.NUM_CLASSES).float()
        label = label.cuda(rank)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(data)
        loss = criterion(output, label)
        adjust_lr(optimizer, epoch, cfg.SOLVER.BASE_LR, cfg)

        # measure accuracy and record loss
        # acc1, acc5 = topks_correct(output, label, (1, 5))
        acc1 = accuracytop1(output, label, (1, ))
        acc5 = accuracytop5(output, label, (5, ))

        # torch.distributed.barrier()

        # reduced_loss = reduce_mean(loss, args.nprocs)
        # reduced_acc1 = reduce_mean(acc1, args.nprocs)
        # reduced_acc5 = reduce_mean(acc5, args.nprocs)

        # losses.update(reduced_loss.item(), data[0].size(0))
        # top1.update(reduced_acc1.item(), data[0].size(0))
        # top5.update(reduced_acc5.item(), data[0].size(0))

        losses.update(loss.item(), data[0].size(0))
        top1.update(acc1.item(), data[0].size(0))
        top5.update(acc5.item(), data[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter % args.print_freq == 0:
            train_message = progress.display(iter)
            logger.info('Train Phase:' + train_message)
    return losses.avg
    


def validate(val_loader, model, criterion, rank, args, logger, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for iter, valdata in enumerate(val_loader):
            data, label = valdata
            if isinstance(data, (list,)):
                for i in range(len(data)):
                    data[i] = data[i].cuda(rank)
            else:
                data = data.cuda(rank)
            label = label.cuda(rank)
            # compute output
            output = model(data)
            loss = criterion(output, label)

            # measure accuracy and record loss
            # acc1, acc5 = topks_correct(output, label, (1, 5))
            acc1 = accuracytop1(output, label, (1, ))
            acc5 = accuracytop5(output, label, (5, ))

            # torch.distributed.barrier()

            # reduced_loss = reduce_mean(loss, args.nprocs)
            # reduced_acc1 = reduce_mean(acc1, args.nprocs)
            # reduced_acc5 = reduce_mean(acc5, args.nprocs)

            # losses.update(reduced_loss.item(), data[0].size(0))
            # top1.update(reduced_acc1.item(), data[0].size(0))
            # top5.update(reduced_acc5.item(), data[0].size(0))

            losses.update(loss.item(), data[0].size(0))
            top1.update(acc1.item(), data[0].size(0))
            top5.update(acc5.item(), data[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iter % args.print_freq == 0:
                val_message = progress.display(iter)
                logger.info('Val Phase:' + val_message)


        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Val Acc@1 {top1.avg:.3f} Val Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg



if __name__ == '__main__':
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args, cfg), join=True)