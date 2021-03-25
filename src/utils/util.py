# -*- coding: UTF-8 -*-

import os
import pathlib
import datetime

from args_and_config.config import config


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
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)

def adjust_lr(optimizer, epoch ,lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 10 epochs
    """
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lst_sum(list):
    sum = 0.0
    for i in range(len(list)):
        sum += list[i]
    return sum

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