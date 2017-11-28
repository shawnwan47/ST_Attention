import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import Utils
import Models

parser = argparse.ArgumentParser(
    description='train_GAT.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--granularity', type=int, default=15)
parser.add_argument('--past', type=int, default=40)
parser.add_argument('--future', type=int, default=8)
# model
parser.add_argument('--nhid', type=int, default=16)
parser.add_argument('--nlay', type=int, default=2)
parser.add_argument('--att_heads', type=int, default=8)
parser.add_argument('--att_reduct', type=str, choices=['cat', 'avg'])
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
parser.add_argument('--savepath', type=str, default='gat.pt')

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
adj = Utils.load_adj()
adj_none = np.ones_like(adj)
adj = torch.LongTensor(adj)
adj_none = torch.LongTensor(adj)


# model
ndim = features_train.shape[-1]
nhid = args.nhid
model = Models.GAT(ndim, args.nhid, args.att_heads, args.future)
print(model.parameters())

if args.gpuid:
    model = model.cuda()
    adj = adj.cuda()
    adj_none = adj_none.cuda()


# training
def trainIters(model, bsz, niter, print_every, lr, lr_min, lr_decay):
    start = time.time()

    loss_all = 0
    loss_best = 1.
    stops = 0

    optimization = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for i in range(1, niter + 1):
        optimization.zero_grad()
        var_inp, var_targ = Utils.get_batch(features_train, labels_train, bsz)
        var_inp = var_inp.transpose(0, 1).transpose(1, 2)
        var_targ = var_targ.transpose(0, 1).transpose(1, 2)

        outs, att1, att2 = model(var_inp, adj)
        loss = criterion(outs, var_targ)
        loss.backward()
        optimization.step()

        loss_all += loss.data[0] / args.future

        if i % print_every == 0:
            loss_avg = loss_all / print_every
            loss_all = 0
            print('iter: %d time: %s lr: %.5f loss: %.4f' % (
                i, Utils.timeSince(start, i / niter), lr, loss_avg))
            # lr decay
            if loss_avg > loss_best:
                stops += 1
                if stops > 3:
                    if lr <= lr_min:
                        return
                    else:
                        lr /= lr_decay
                        stops = 0
            else:
                loss_best = loss_avg
                stops = 0


trainIters(model,
           args.bsz, args.niter, args.print_every,
           args.lr, args.lr_min, args.lr_decay)

torch.save(model, args.savepath)
