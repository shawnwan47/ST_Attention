import argparse
import pickle

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

import Args
import Models
from Loss import Loss
import Utils

args = argparse.ArgumentParser()
Args.add_data(args)
Args.add_model(args)
Args.add_train(args)
args = args.parse_args()
Args.update(args)
print(args)

# CUDA
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpuid)
    float_type = torch.cuda.FloatTensor
    long_type = torch.cuda.LongTensor
    torch.cuda.manual_seed(args.seed)
    print(f'Using GPU: {args.gpuid}')
else:
    torch.manual_seed(args.seed)
    float_type = torch.FloatTensor
    long_type = torch.LongTensor
    print('Using CPU')

# DATA
data_train, data_valid, data_test, mean, scale = Utils.getDataset(
    dataset=args.dataset,
    freq=args.freq,
    start=args.start,
    past=args.past,
    future=args.future,
    batch_size=args.batch_size,
    cuda=args.cuda
)


# MODEL
model_path = 'model/' + args.path
if args.test or args.retrain:
    model = torch.load(model_path + '.pt')
else:
    model = getattr(Models, args.model)(args)
if args.cuda:
    model.cuda()

# LOSS
loss = Loss(mean, scale, args.output_od)

# OPTIM
if args.optim is 'SGD':
    optimizer = optim.SGD(model.parameters(),
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=True)
else:
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)


# TRAINING
def train_model(data):
    model.train()
    mse_avg = wape_avg = iters = 0
    for _ in range(args.iterations):
        for data_num, data_cat, target in data:
            data_num = Variable(data_num).type(float_type)
            data_cat = Variable(data_cat).type(long_type)
            target = Variable(target).type(float_type)
            output = model(data_num, data_cat)
            if type(output) is tuple:
                output = output[0]
            mse, wape = loss(output, target)
            mse_avg += mse.data[0]
            wape_avg += wape.data[0]
            # optimization
            optimizer.zero_grad()
            if args.loss is 'mse':
                mse.backward()
            elif args.loss is 'wape':
                wape.backward()
            clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
            iters += 1
    return mse_avg / iters, wape_avg / iters


def eval_model(data):
    model.eval()
    mse_avg = wape_avg = iters = 0
    infos = []
    for data_num, data_cat, target in data:
        data_num = Variable(data_num, volatile=True).type(float_type)
        data_cat = Variable(data_cat, volatile=True).type(long_type)
        target = Variable(target, volatile=True).type(float_type)
        output = model(data_num, data_cat)
        if type(output) is tuple:
            output, more = output[0], output[1:]
            infos.append(more)
        mse, wape = loss(output, target)
        mse_avg += mse.data[0]
        wape_avg += wape.data[0]
        iters += 1
    if infos:
        infos = [torch.cat(info, 0).cpu().data.numpy() for info in zip(*infos)]
    return mse_avg / iters, wape_avg / iters, infos


if not args.test:
    for epoch in range(args.epoches):
        loss_train, wape_train = train_model(data_train)
        loss_valid, wape_valid, _ = eval_model(data_valid)

        print(f'{epoch}\t'
              f'loss:{loss_train:.4f} {loss_valid:.4f}\t'
              f'wape:{wape_train:.4f} {wape_valid:.4f}')

        scheduler.step(loss_valid)
        if optimizer.param_groups[0]['lr'] < args.min_lr:
            break

loss_test, wape_test, info = eval_model(data_test)
print(f'Test loss:{loss_test:.4f} wape:{wape_test:.4f}')
model.cpu()
torch.save(model, model_path + '.pt')
pickle.dump(info, open(model_path + '.pkl', 'wb'))
