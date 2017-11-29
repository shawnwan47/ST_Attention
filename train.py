import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import Data
import Models

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--gran', type=int, default=15)
parser.add_argument('--past', type=int, default=8)
parser.add_argument('--future', type=int, default=4)
parser.add_argument('--raw_flow', action='store_true')
# model
parser.add_argument('--nhid', type=int, default=512)
parser.add_argument('--nlay', type=int, default=1)
parser.add_argument('--pdrop', type=float, default=0.1)
parser.add_argument('--attn', action='store_true')
parser.add_argument('--attn_type', type=str, default='general',
                    choices=['dot', 'general', 'concat'])
parser.add_argument('--loss', type=str, default='MAPE',
                    choices=['WAPE', 'MAPE'])
parser.add_argument('--weight_decay', type=float, default=0.00001)
# train
parser.add_argument('--bsz', type=int, default=1000)
parser.add_argument('--niter', type=int, default=10000)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--iprint', type=int, default=100)
parser.add_argument('--lr', type=float, default=1.)
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
inputs_train = Variable(torch.FloatTensor(inputs_train))
targets_train = Variable(torch.FloatTensor(targets_train))
inputs_valid = Variable(torch.FloatTensor(inputs_valid)).transpose(0, 1)
targets_valid = Variable(torch.FloatTensor(targets_valid)).transpose(0, 1)
if args.gpuid:
    inputs_train = inputs_train.cuda()
    targets_train = targets_train.cuda()
    inputs_valid = inputs_valid.cuda()
    targets_valid = targets_valid.cuda()
dataset = TensorDataset(inputs_train, targets_train)
dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=True)

# model
ndim = inputs_train.size(-1)
model = Models.seq2seq(args.past, args.future,
                       ndim, args.nhid, args.nlay, args.pdrop,
                       args.attn)
if args.gpuid:
    model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# training
for epoch in range(args.nepoch):
    # train
    loss_train = []

    for inputs, targets in dataloader:
        inputs = inputs.transpose(0, 1)
        targets = targets.transpose(0, 1)
        if args.attn:
            outputs, attns = model(inputs, targets, args.gpuid, teach=True)
        else:
            outputs = model(inputs, targets, args.gpuid, teach=True)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_train.append(loss.data[0] / args.future)

    loss_train = sum(loss_train) / len(loss_train)
    if args.attn:
        outputs, attns = model(inputs_valid, targets_valid, args.gpuid)
    else:
        outputs = model(inputs_valid, targets_valid, args.gpuid)
    loss_valid = criterion(outputs, targets_valid).data[0]
    print('%d train: %.4f valid: %.4f' % (epoch, loss_train, loss_valid))

    scheduler.step(loss_valid)


torch.save(model, args.savepath)
