import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Utils import load_flow, split_dataset, np2torch
from Models import EncoderRNN, AttnDecoderRNN
from Trainer import seq2seq_attn
from Constants import USE_CUDA


# CUDA
torch.manual_seed(47)
if torch.cuda.is_available() and USE_CUDA:
    torch.cuda.manual_seed(47)

# data
features, labels, days, times, flow_mean, flow_std = load_flow()

enc = torch.load('enc.pk')
dec = torch.load('dec.pk')

var_features = np2torch(features[:64])
var_labels = np2torch(labels[:64])

dec_outs, dec_attns = seq2seq_attn(var_features, enc, dec)

np.save('attns', dec_attns.numpy())
