import argparse

import numpy as np

import torch
import torch.optim as optim
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
Args.update(args)
print(args)

# CUDA
if args.gpuid:
    torch.cuda.set_device(args.gpuid)
if args.seed > 0:
    torch.cuda.manual_seed(args.seed)

# DATA
data_train, data_valid, data_test, mean, scale = Utils.getDataset(
    dataset=args.dataset,
    freq=args.freq,
    start=args.start,
    past=args.past,
    future=args.future,
    batch_size=args.batch_size
)


def WAPE(out, tgt):
    out, tgt = denormalize(out), denormalize(tgt)
    return (tgt - out).abs().sum() / tgt.sum()


def denormalize(data):
    return data * scale + mean

# MODEL
modelpath = MODEL_PATH + args.path
print('Model: {}'.format(modelpath))

model = getattr(Models, args.model)(args)
if args.test:
    model = torch.load(modelpath + '.pt')
    print('Model loaded.')
model.cuda()

# LOSS
criterion = getattr(torch.nn, args.crit)()

# OPTIM
if args.optim is 'SGD':
    optimizer = optim.SGD(model.parameters(),
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=True)
else:
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=args.min_lr)


# TRAINING
def train_model(data):
    model.train()
    loss_train = wape = iters = 0
    for _ in range(args.iterations):
        for data_num, data_cat, targets in data:
            data_num = Variable(data_num).cuda()
            data_cat = Variable(data_cat).cuda()
            targets = Variable(targets).cuda()
            out, _ = model(data_num, data_cat)
            loss = criterion(out, targets)
            loss_train += loss.data[0]
            wape += WAPE(out, targets).data[0]
            iters += 1
            # optimization
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
    return loss_train / iters, wape / iters


def eval_model(data):
    model.eval()
    loss = wape = iters = 0
    atts = []
    for data_num, data_cat, targets in data:
        data_num = Variable(data_num, volatile=True).cuda()
        data_cat = Variable(data_cat, volatile=True).cuda()
        targets = Variable(targets, volatile=True).cuda()
        out, att = model(data_num, data_cat)
        loss += criterion(out, targets).data[0]
        wape += WAPE(out, targets).data[0]
        iters += 1
        atts.append(att)
    return loss / iters, wape / iters, torch.cat(atts, 0)


if not args.test:
    for epoch in range(args.epoches):
        loss_train, wape_train = train_model(data_train)
        loss_valid, wape_valid, _ = eval_model(data_valid)
        loss_test, wape_test, att = eval_model(data_test)

        print(f'{epoch:1} {loss_train:.4f} {wape_train:.4f} {loss_valid:.4f} {wape_valid:.4f} {loss_test:.4f} {wape_test:.4f}')

        scheduler.step(loss_valid)
        if optimizer.param_groups[0]['lr'] < args.min_lr:
            break

torch.save(model.cpu(), modelpath + '.pt')
Utils.torch2npsave(modelpath + '_att', att)
