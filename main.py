import argparse

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

import Args
import Models
import Utils
from Consts import MODEL_PATH


args = argparse.ArgumentParser('Traffic Forecasting')
Args.add_gpu(args)
Args.add_data(args)
Args.add_loss(args)
Args.add_optim(args)
Args.add_model(args)
args = args.parse_args()

# CUDA
if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
if args.seed > 0:
    torch.cuda.manual_seed(args.seed)

# DATA
(flow_train, flow_valid, flow_test,
 daytime_train, daytime_valid, daytime_test,
 flow_mean, flow_std) = Utils.load_data(args)

flow_train = Variable(flow_train)
flow_valid = Variable(flow_valid, volatile=True)
flow_test = Variable(flow_test, volatile=True)
daytime_train = Variable(daytime_train)
daytime_valid = Variable(daytime_valid, volatile=True)
daytime_test = Variable(daytime_test, volatile=True)
flow_mean = Variable(flow_mean, requires_grad=False)
flow_std = Variable(flow_std, requires_grad=False)

past, future = args.past, args.future
if args.yesterday:
    past += flow_train.size(0) // 2
if args.daytime:
    args.input_size += args.day_size + args.time_size


def denormalize(flow):
    return flow * flow_std + flow_mean


# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))
model = getattr(Models, args.model)(args).cuda()
daytime = Models.DayTime(args).cuda()

# LOSS
criterion = getattr(torch.nn, args.loss)()

# OPTIM
parameters = list(model.parameters()) + list(daytime.parameters())
optimizer = getattr(torch.optim, args.optim)(
    parameters, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.patience, verbose=True)


def train_model():
    loss_train = []
    for t in range(past, flow_train.size(0) - future - 1):
        src = flow_train[:t]
        tgt = flow_train[t + 1:t + future + 1]
        inp = flow_train[t:t + future]
        if args.daytime:
            src = torch.cat((src, daytime(daytime_train[:t])), -1)
            inp = torch.cat((inp, daytime(daytime_train[t:t + future])), -1)
        out = model(src, inp, teach=True)
        out = denormalize(out[0])
        tgt = denormalize(tgt)
        loss = criterion(out, tgt)
        loss.backward()
        clip_grad_norm(parameters, args.max_grad_norm)
        optimizer.step()
        loss_train.append(loss.data[0])
    return sum(loss_train) / len(loss_train)


def valid_model():
    loss_valid = []
    for t in range(past, flow_valid.size(0) - future - 1):
        src = flow_valid[:t]
        tgt = flow_valid[t + 1:t + future + 1]
        inp = flow_valid[t:t + future]
        if args.daytime:
            src = torch.cat((src, daytime(daytime_valid[:t])), -1)
            inp = torch.cat((inp, daytime(daytime_valid[t:t + future])), -1)
        out = model(src, inp, teach=False)
        out = denormalize(out[0])
        tgt = denormalize(tgt)
        loss = criterion(out, tgt)
        loss_valid.append(loss.data[0])
    return sum(loss_valid) / len(loss_valid)


def eval_model():
    def pe(out, tgt):
        return criterion(out, tgt).data[0] / tgt.mean()

    times, days = flow_test.size()[:2]
    end = times - future - 1
    outs = []
    tgts = []
    for t in range(past, end):
        src = flow_test[:t]
        tgt = flow_test[t + 1:t + future + 1]
        inp = flow_test[t:t + future]
        if args.daytime:
            src = torch.cat((src, daytime(daytime_test[:t])), -1)
            inp = torch.cat((inp, daytime(daytime_test[t:t + future])), -1)
        out = model(src, inp, teach=False)
        out = denormalize(out[0])
        tgt = denormalize(tgt)
        outs.append(out)
        tgts.append(tgt)
    outs = torch.stack(outs, 1)
    tgts = torch.stack(tgts, 1)
    loss = [pe(outs[i], tgts[i]) for i in range(args.future)]
    return np.array(list(map(float, loss)))


def attn_model():
    src = flow_test[:past]
    tgt = flow_test[past:-1]
    if args.daytime:
        src = torch.cat((src, daytime(daytime_test[:past])), -1)
        tgt = torch.cat((tgt, daytime(daytime_test[past:-1])), -1)
    out = model(src, tgt, teach=True)
    attn = out[-1].data.numpy()
    return attn


# TRAINING
if not args.test:
    for epoch in range(args.nepoch):
        loss_train = train_model()
        loss_valid = valid_model()

        print('Epoch: %d train: %.4f valid: %.4f' % (
            epoch, loss_train, loss_valid))

        scheduler.step(loss_valid)

    torch.save(model.cpu(), modelpath + '.pt')

# TESTING
model = torch.load(modelpath + '.pt').cuda()

loss_test = eval_model()
print('Test {}: {}'.format(modelpath, loss_test))
np.savetxt(modelpath + '_loss.txt', loss_test)

if (args.model == 'RNN' and args.attn) or args.model == 'Transformer':
    attn = attn_model()
    np.save(modelpath + '_attn', attn)
