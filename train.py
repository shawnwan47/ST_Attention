import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.utils.data import TensorDataset, DataLoader

import Data
import Models
import Loss

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('-gran', type=int, default=15)
parser.add_argument('-past', type=int, default=32)
parser.add_argument('-future', type=int, default=8)
parser.add_argument('-raw_flow', action='store_true')
# model
parser.add_argument('-nhid', type=int, default=1000)
parser.add_argument('-nlay', type=int, default=1)
parser.add_argument('-pdrop', type=float, default=0.1)
parser.add_argument('-attn', action='store_true')
parser.add_argument('-attn_type', type=str, default='general',
                    choices=['dot', 'general', 'concat'])
parser.add_argument('-loss', type=str, default='MSELoss',
                    choices=['WAPE', 'MAPE'])
parser.add_argument('-weight_decay', type=float, default=0.00001)
# train
parser.add_argument('-bsz', type=int, default=1000)
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

# data
inputs, targets, days, times, data_mean, data_std = Data.load_flow(
    gran=args.gran, past=args.past, future=args.future, raw=args.raw_flow)
inputs_train, inputs_valid, inputs_test = Data.split_dataset(inputs)
targets_train, targets_valid, targets_test = Data.split_dataset(targets)
inputs_train = Variable(torch.FloatTensor(inputs_train)).transpose(0, 1)
targets_train = Variable(torch.FloatTensor(targets_train)).transpose(0, 1)
inputs_valid = Variable(torch.FloatTensor(inputs_valid)).transpose(0, 1)
targets_valid = Variable(torch.FloatTensor(targets_valid)).transpose(0, 1)
data_mean = Variable(torch.FloatTensor(data_mean))
data_std = Variable(torch.FloatTensor(data_std))
if args.gpuid:
    inputs_train = inputs_train.cuda()
    targets_train = targets_train.cuda()
    inputs_valid = inputs_valid.cuda()
    targets_valid = targets_valid.cuda()
    data_mean = data_mean.cuda()
    data_std = data_std.cuda()
# dataset = TensorDataset(inputs_train, targets_train)
# dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=True)

# model
ndim = inputs_train.size(-1)
model = Models.seq2seq(args.past, args.future,
                       ndim, args.nhid, args.nlay, args.pdrop,
                       args.attn)
if args.gpuid:
    model = model.cuda()

criterion = Loss.WAPE

loss_min = float('inf')
lr_stop = 0

# training
for epoch in range(args.nepoch):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_train = []
    for _ in range(args.niter):
        if args.attn:
            outputs, attns = model(inputs_train, targets_train, args.gpuid, True)
        else:
            outputs = model(inputs_train, targets_train, args.gpuid, True)
        loss = criterion(outputs, targets_train)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data[0])
    loss_train = sum(loss_train) / len(loss_train)

    if args.attn:
        outputs, attns = model(inputs_valid, targets_valid, args.gpuid)
    else:
        outputs = model(inputs_valid, targets_valid, args.gpuid)
    loss_valid = criterion(outputs, targets_valid).data[0]
    outputs = outputs * data_std + data_mean
    targets_valid = targets_valid * data_std + data_mean
    wape = Loss.WAPE(outputs, targets_valid).data[0]
    mape = Loss.MAPE(outputs, targets_valid).data[0]
    print('%d lr: %.e train: %.4f valid: %.4f WAPE: %.4f MAPE: %.4f' % (
        epoch, args.lr, loss_train, loss_valid, wape, mape))

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

# testing
if args.attn:
    outputs, attns = model(inputs_test, targets_test, args.gpuid)
    np.save('attns', attns.numpy())
else:
    outputs = model(inputs_test, targets_test, args.gpuid)

loss = criterion(outputs, targets_test)
outputs = outputs * data_std + data_mean
targets_test = targets_test * data_std + data_mean
wape = Loss.WAPE(outputs, targets_test).data[0]
mape = Loss.MAPE(outputs, targets_test).data[0]

print('MSE: %.4f WAPE: %.4f MAPE: %.4f' % (loss.data[0], wape, mape))


# result
def cuda2np(x):
    return x.cpu().data.numpy()


np.save('test_results', [cuda2np(targets_test), cuda2np(outputs)])
