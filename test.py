import argparse

import numpy as np

import torch
from torch.autograd import Variable

import Data
import Utils
import Loss
import Consts


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--gran', type=int, default=15)
parser.add_argument('--past', type=int, default=16)
parser.add_argument('--future', type=int, default=4)
parser.add_argument('--raw_flow', action='store_true')
# gpu
parser.add_argument('--gpuid', type=list, default=[3])
parser.add_argument('--seed', type=int, default=47)
# file
parser.add_argument('--savepath', type=str, default='seq2seq.pt')

args = parser.parse_args()

if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

if args.attn:
    print('Training seq2seq with Attention')
else:
    print('Training seq2seq without Attention')

# DATA
inputs, targets, days, times, data_mean, data_std = Data.load_flow(
    gran=args.gran, past=args.past, future=args.future)
inputs = torch.FloatTensor(inputs)
targets = torch.FloatTensor(targets)
data_mean = Variable(torch.FloatTensor(data_mean))
data_std = Variable(torch.FloatTensor(data_std))
if args.gpuid:
    inputs = inputs.cuda()
    targets = targets.cuda()
    data_mean = data_mean.cuda()
    data_std = data_std.cuda()
unit = inputs.size(0) // Consts.DAYS
_, _, inputs = Data.split_dataset(inputs, unit)
_, _, targets = Data.split_dataset(targets, unit)

inputs = Utils.tensor2VarRNN(inputs)
targets = Utils.tensor2VarRNN(inputs)
targets_raw = targets * data_std + data_mean

# model
model = torch.load(args.savepath)
outputs = model(inputs, targets, args.gpuid)
if model.attn:
    outputs, attns = outputs
    np.save('attns', attns.numpy())

outputs = outputs * data_std + data_mean
wape = Loss.WAPE(outputs, targets_raw).data[0]
mape = Loss.MAPE(outputs, targets_raw).data[0]

print('WAPE: %.4f MAPE: %.4f' % (wape, mape))

np.save('test_results', [Utils.cuda2np(targets), Utils.cuda2np(outputs)])
