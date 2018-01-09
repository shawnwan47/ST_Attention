import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

import Args
import Models
import Utils
from Consts import MODEL_PATH


args = argparse.ArgumentParser()
Args.add_gpu(args)
Args.add_data(args)
Args.add_loss(args)
Args.add_optim(args)
Args.add_run(args)
Args.add_model(args)
args = args.parse_args()
Args.update_args(args)
print(args)

# CUDA
if args.gpuid:
    torch.cuda.set_device(args.gpuid)
if args.seed > 0:
    torch.cuda.manual_seed(args.seed)

# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))

model = getattr(Models, args.model)(args)
if args.test:
    model = torch.load(modelpath + '.pt')
    print('Loaded models from file.')
model.cuda()

# DATA
(inp_train, inp_valid, inp_test,
 tgt_train, tgt_valid, tgt_test,
 st_train, st_valid, st_test,
 flow_min, flow_scale) = Utils.load_data(args)


def WAPE(out, tgt):
    '''
    out: batch x future x loc x weight
    tgt: batch x future x loc
    '''
    _, out = out.max(1)
    out = out.type(torch.cuda.FloatTensor)
    tgt = tgt.type(torch.cuda.FloatTensor)
    out = out * flow_scale + flow_min
    tgt = tgt * flow_scale + flow_min
    wape = torch.abs(out - tgt).sum() / torch.sum(tgt)
    return wape.data[0]


def WAPE_(out, tgt):
    out = out.type(torch.cuda.FloatTensor)
    tgt = tgt.type(torch.cuda.FloatTensor)
    out = out * flow_scale + flow_min
    tgt = tgt * flow_scale + flow_min
    wape = torch.abs(out - tgt).sum() / torch.sum(tgt)
    return wape.data[0]


# LOSS
criterion = getattr(torch.nn, args.loss)()

# OPTIM
optimizer = getattr(torch.optim, args.optim)(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)


# TRAINING
def train():
    model.train()
    loss_train = wape = 0
    num_sample = inp_train.size(0)
    days = torch.randperm(num_sample)
    iters = num_sample // args.batch
    for day in range(iters):
        idx = days[day::iters]
        inp = inp_train[idx].cuda()
        tgt = tgt_train[idx].cuda()
        st = st_train[idx].cuda()
        out, _ = model(inp, st)
        loss = criterion(out, tgt)
        loss_train += loss.data[0]
        wape += WAPE(out, tgt)
        # optimization
        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return loss_train / iters, wape / iters


def valid():
    model.eval()
    loss = wape = 0
    days = inp_valid.size(0) // args.num_time
    for i in range(days):
        inp = inp_valid[i * args.num_time:(i + 1) * args.num_time].cuda()
        tgt = tgt_valid[i * args.num_time:(i + 1) * args.num_time].cuda()
        st = st_valid[i * args.num_time:(i + 1) * args.num_time].cuda()
        out, _ = model(inp, st)
        loss += criterion(out, tgt).data[0]
        wape += WAPE(out, tgt)
    return loss / days, wape / days


def test():
    model.eval()
    loss = wape = 0
    days = inp_test.size(0) // args.num_time
    atts = []
    for i in range(days):
        inp = inp_test[i * args.num_time:(i + 1) * args.num_time].cuda()
        tgt = tgt_test[i * args.num_time:(i + 1) * args.num_time].cuda()
        st = st_test[i * args.num_time:(i + 1) * args.num_time].cuda()
        out, att = model(inp, st)
        loss += criterion(out, tgt).data[0]
        wape += WAPE(out, tgt)
        atts.append(att)
    return loss / days, wape / days, torch.stack(atts, 0)


if not args.test:
    for epoch in range(args.epoches):
        loss_train, wape_train = train()
        loss_valid, wape_valid = valid()

        if not epoch % args.print_epoches:
            print('Epoch: %d NLL: %.4f %.4f WAPE: %.4f %.4f' % (
                epoch, loss_train, loss_valid, wape_train, wape_valid))

        scheduler.step(loss_valid)
        if optimizer.param_groups[0]['lr'] < args.lr_min:
            break

loss_test, wape_test, att = test()
print('Test {}: NLL:{} WAPE:{}'.format(modelpath, loss_test, wape_test))
torch.save(model.cpu(), modelpath + '.pt')
Utils.torch2npsave(modelpath + '_att', att)
