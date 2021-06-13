# -*- coding: UTF-8 -*-

import os
import sys
import pathlib
import datetime
import builtins
import shutil
import logging
import numpy as np

import torch
import torch.distributed as dist


def compute_root_dir():
    root_dir = os.path.abspath(
        os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__))))
    return root_dir + os.path.sep


proj_root_dir = pathlib.Path(compute_root_dir())


def file_exists(file_path):
    return os.path.exists(file_path)


def compute_time_span(start_time, end_time, date_format="%Y/%m/%d %H:%M"):
    new_start_time = datetime.datetime.strptime(start_time, date_format)
    new_end_time = datetime.datetime.strptime(end_time, date_format)

    span = new_end_time - new_start_time
    return span

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # 使用type_as(tesnor)将张量转换为给定类型的张量。
    labels = torch.max(labels, 1)[1]
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)

def topks_correct(preds, labels, ks):

    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    labels = torch.max(labels, 1)[1].numpy()
    labels = labels.view(-1,1)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    topks_score = [(x / preds.size(0)) * 100.0 for x in topks_correct]
    return topks_score[0], topks_score[1]

def accuracytop1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res[0]

def accuracytop5(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res[0]

def adjust_lr(optimizer, epoch ,lr, cfg):
    """
    Sets the learning rate to the initial LR decayed by 10 every 10 epochs
    """
    lr = lr * (cfg.SOLVER.LR_DECAY ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_weight(labels, n_classes):
    num_label = []
    weight_per_class = []
    weight = [0] * len(labels)
    for i in range(n_classes):
        num_label.append(sum(labels==i))
    N = float(sum(num_label))
    for label in num_label:
        weight_per_class.append(N/float(label))
    for idx, value in enumerate(labels):
        weight[idx] = weight_per_class[value]
    return weight

def is_master_proc(args):
    if torch.distributed.is_initialized():
        return dist.get_rank() % args.num_gpus == 0
    else:
        return True

def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.pth')

# log helper
def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass

def get_logger(args):
    logger = logging.getLogger(__name__)

    logger.setLevel(level=logging.INFO)
    logger.propagate = False
    if not os.path.exists(args.log_path):
        f = open(args.log_path, 'a')
        f.close()
    if is_master_proc(args):
        handler = logging.FileHandler(args.log_path, encoding='utf-8')
        handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(level=logging.INFO)

        logger.addHandler(console)
        logger.addHandler(handler)
    else:
        _suppress_print()

    return logger