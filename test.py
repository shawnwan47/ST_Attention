import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import Utils
from Models import EncoderRNN, AttnDecoderRNN
from Trainer import seq2seq_attn

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--granularity', type=int, default=15)
parser.add_argument('--past', type=int, default=40)
parser.add_argument('--future', type=int, default=8)
# model
parser.add_argument('--nhid', type=int, default=512)
parser.add_argument('--nlay', type=int, default=2)
parser.add_argument('--attn_type', type=str, default='general',
                    choices=['dot', 'general', 'concat'])
parser.add_argument('--loss', type=str, default='MAPE',
                    choices=['WAPE', 'MAPE'])
# train
parser.add_argument('--bsz', type=int, default=100)
parser.add_argument('--niter', type=int, default=10000)
parser.add_argument('--nepoch', type=int, default=10)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_min', type=float, default=0.00001)
parser.add_argument('--lr_decay', type=int, default=5)
# gpu
parser.add_argument('--gpuid', type=list, default=[3])
parser.add_argument('--seed', type=int, default=47)
# file
parser.add_argument('--savepath', type=str, default='save.pt')

args = parser.parse_args()

if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

# data
features, labels, days, times, flow_mean, flow_std = Utils.load_flow(
    granularity=args.granularity, past=args.past, future=args.future)
features_train, features_valid, features_test = Utils.split_dataset(features)
labels_train, labels_valid, labels_test = Utils.split_dataset(labels)
oneday = features_test.shape[0] // 4
var_inp = Utils.np2torch(features_test[-oneday:])
var_out = Utils.np2torch(labels_test[-oneday:])

# model
encoder, decoder = torch.load('save.pt')

outs, attns = seq2seq_attn(var_inp, var_out, encoder, decoder)

# result
np.save('attns', attns.numpy())
np.save('predictions', [var_out.cpu().data.numpy(), outs.cpu().data.numpy()])
