import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import Data
import Models
import Loss
import Utils
import Consts

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('-flow_type', type=str, default='seq')
parser.add_argument('-gran', type=int, default=15)
parser.add_argument('-past', type=int, default=8)
parser.add_argument('-future', type=int, default=4)
# model
parser.add_argument('-nhid', type=int, default=600)
parser.add_argument('-nlay', type=int, default=1)
parser.add_argument('-pdrop', type=float, default=0.5)
parser.add_argument('-attn', action='store_true')
parser.add_argument('-attn_type', type=str, default='general',
                    choices=['dot', 'general', 'concat'])
parser.add_argument('-weight_decay', type=float, default=0.00001)
# train
parser.add_argument('-bsz', type=int, default=100)
parser.add_argument('-nepoch', type=int, default=100)
parser.add_argument('-niter', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.1)
parser.add_argument('-lr_min', type=float, default=1e-8)
parser.add_argument('-lr_decay', type=float, default=0.1)
parser.add_argument('-lr_patience', type=int, default=5)
# gpu
parser.add_argument('-gpuid', default=[], nargs='+', type=int)
parser.add_argument('-seed', type=int, default=47)
# file
parser.add_argument('-savepath', type=str, default='seq2seq.pt')

args = parser.parse_args()
print(args)

if torch.cuda.is_available() and not args.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

if args.attn:
    print('Training seq2seq with Attention')
else:
    print('Training seq2seq without Attention')

# DATA
(flow_train, flow_valid, flow_test,
 days_train, days_valid, days_test,
 times_train, times_valid, times_test,
 flow_mean, flow_std) = Utils.load_data(args)
flow_valid_raw = flow_valid * flow_std + flow_mean

# MODEL
ndim = inputs_train.size(-1)
model = Models.seq2seq(args.past, args.future,
                       ndim, args.nhid, args.nlay, args.pdrop,
                       args.attn)
if args.gpuid:
    model = model.cuda()

criterion = nn.MSELoss()

loss_min = float('inf')
lr_stop = 0

for epoch in range(args.nepoch):
    # train
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_train = []
    for _ in range(args.niter):
        for inputs, targets in dataloader:
            inputs = Utils.tensor2VarRNN(inputs)
            targets = Utils.tensor2VarRNN(targets)
            outputs = model(inputs, targets, args.gpuid, True)
            if type(outputs) is tuple:
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_train.append(loss.data[0])
    loss_train = sum(loss_train) / len(loss_train)

    # valid
    outputs = model(inputs_valid, flow_valid, args.gpuid)
    if type(outputs) is tuple:
        outputs = outputs[0]
    loss_valid = criterion(outputs, flow_valid).data[0]
    outputs = outputs * flow_std + flow_mean
    wape = Loss.WAPE(outputs, flow_valid_raw).data[0]
    mape = Loss.MAPE(outputs, flow_valid_raw).data[0]
    print('%d lr: %.e train: %.4f valid: %.4f WAPE: %.4f MAPE: %.4f' % (
        epoch, args.lr, loss_train, loss_valid, wape, mape))

    # lr schedule
    if loss_valid > loss_min:
        lr_stop += 1
        if lr_stop == args.lr_patience:
            if args.lr <= args.lr_min:
                break
            args.lr *= args.lr_decay
            lr_stop = 0
    else:
        loss_min = loss_valid
        lr_stop = 0

torch.save(model, args.savepath)
