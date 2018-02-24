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
Args.add_data(args)
Args.add_model(args)
Args.add_train(args)
args = args.parse_args()
Args.update_args(args)
print(args)

# CUDA
if args.gpuid:
    torch.cuda.set_device(args.gpuid)
if args.seed > 0:
    torch.cuda.manual_seed(args.seed)

# DATA
data_train, data_valid, data_test, mean, std = Utils.getDataset(
    dataset=args.dataset,
    freq=args.freq,
    start=args.start,
    past=args.past,
    future=args.future
)
num_loc = data_test[0].size(-2)

# MODEL
modelpath = MODEL_PATH + args.path
print('Model: {}'.format(modelpath))

model = getattr(Models, args.model)(args)
if args.test or args.retrain:
    model = torch.load(modelpath + '.pt')
    print('Model loaded.')
model.cuda()

# LOSS
criterion = getattr(torch.nn, args.crit)

# OPTIM
optimizer = getattr(torch.optim, args.optim)(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)


# TRAINING
def train(inputs, targets):
    model.train()
    loss_train = wape = 0
    num_sample = inp_train.size(0)
    iters = num_sample // args.batch
    for _ in range(args.iterations):
        random_idx = torch.randperm(num_sample)
        for i in range(iters):
            idx = random_idx[i::iters]
            inp = Variable(inputs[idx]).cuda()
            tgt = Variable(targets[idx]).cuda()
            flow = Variable(targets[idx][:, :, :, 0]).cuda()
            out, _ = model(inp, tgt)
            loss = criterion(out, flow)
            loss_train += loss.data[0]
            wape += WAPE(out, flow)
            # optimization
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
    iters *= args.iterations
    return loss_train / iters, wape / iters


def eval(inputs, targets):
    model.eval()
    loss = wape = 0
    days = inputs.size(0) // args.num_time
    atts = []
    for i in range(days):
        inp = Variable(inputs[i * args.num_time:(i + 1) * args.num_time], volatile=True).cuda()
        tgt = Variable(targets[i * args.num_time:(i + 1) * args.num_time], volatile=True).cuda()
        flow = Variable(targets[i * args.num_time:(i + 1) * args.num_time][:, :, :, 0], volatile=True).cuda()
        out, att = model(inp, tgt)
        loss += criterion(out, flow).data[0]
        wape += WAPE(out, flow)
        atts.append(att)
    return loss / days, wape / days, torch.stack(atts, 0)


if not args.test:
    for epoch in range(args.epoches):
        loss_train, wape_train = train(inp_train, tgt_train)
        loss_valid, wape_valid, _ = eval(inp_valid, tgt_valid)

        print('Epoch: %d NLL: %.4f %.4f WAPE: %.4f %.4f' % (
            epoch, loss_train, loss_valid, wape_train, wape_valid))

        scheduler.step(loss_valid)
        if optimizer.param_groups[0]['lr'] < args.lr_min:
            break

loss_test, wape_test, att = eval(inp_test, tgt_test)
print('Test {}: NLL:{} WAPE:{}'.format(modelpath, loss_test, wape_test))
torch.save(model.cpu(), modelpath + '.pt')
Utils.torch2npsave(modelpath + '_att', att)
