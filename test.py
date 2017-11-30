import argparse

import numpy as np

import torch
from torch.autograd import Variable

import Data


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--gran', type=int, default=20)
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

# data
inputs, targets, days, times, flow_mean, flow_std = Data.load_flow(
    gran=args.gran, past=args.past, future=args.future, raw=args.raw_flow)
inputs_train, inputs_valid, inputs_test = Data.split_dataset(inputs)
targets_train, targets_valid, targets_test = Data.split_dataset(targets)

inputs = torch.FloatTensor(inputs_test)
targets = torch.FloatTensor(targets_test)
inputs = Variable(inputs).transpose(0, 1)
targets = Variable(targets).transpose(0, 1)
if args.gpuid:
    inputs = inputs.cuda()
    targets = targets.cuda()

# model
model = torch.load(args.savepath)
if model.attn:
    outputs, attns = model(inputs, targets, args.gpuid)
    np.save('attns', attns.numpy())
else:
    outputs = model(inputs, targets, args.gpuid)


# result
def cuda2np(x):
    return x.cpu().data.numpy()


np.save('test_results', [cuda2np(targets), cuda2np(outputs)])
