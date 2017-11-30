import time
import math

import torch
from torch.autograd import Variable


def tensor2VarRNN(t):
    return Variable(t.transpose(0, 1))


def var2np(x):
    return x.cpu().data.numpy()
