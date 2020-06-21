import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import logging
import time
import torch.nn.functional as F
import sys

def softmax(input, mask=None, beta=1.0):

    b = input.shape[0]
    exp_A_ = torch.exp(input)
    if mask is not None:
        exp_A_ = mask * exp_A_ # (b, n)
    exp_sum = torch.sum(exp_A_, 1) # (b)
    if beta != 1.0:
        exp_sum = torch.pow(exp_sum, beta)

    tmp_sum = torch.ones_like(exp_sum)*0.000000001
    exp_sum = torch.max(exp_sum, tmp_sum)

    A = (exp_A_/ exp_sum.view(b, -1)) # (b, n)  attention weights
    return A

def readlines(infile):
    with open(infile, 'r') as fin:
        for l in fin:
            yield l.strip()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def copy_model(new_model, original_model):
    new_paras = list(new_model.parameters())
    original_paras = list(original_model.parameters())
    for i in range(len(original_paras)):
        new_paras[i].data[:] = original_paras[i].data[:]

    return
