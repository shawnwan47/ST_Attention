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


args = argparse.ArgumentParser('Traffic Forecasting')
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
    torch.cuda.set_device(args.gpuid[0])
if args.seed > 0:
    torch.cuda.manual_seed(args.seed)

# DATA
data = getattr(Utils, 'load_data_' + args.data_type)(args)
(inp_train, inp_valid, inp_test,
 tgt_train, tgt_valid, tgt_test,
 dt_train, dt_valid, dt_test,
 flow_mean, flow_std, adj) = data


inp_train = Variable(inp_train)
inp_valid = Variable(inp_valid, volatile=True)
inp_test = Variable(inp_test, volatile=True)
dt_train = Variable(dt_train)
dt_valid = Variable(dt_valid, volatile=True)
dt_test = Variable(dt_test, volatile=True)
tgt_train = Variable(tgt_train)
tgt_valid = Variable(tgt_valid, volatile=True)
tgt_test = Variable(tgt_test, volatile=True)
flow_mean = Variable(flow_mean)
flow_std = Variable(flow_std)
adj = Variable(adj)
if not args.adj:
    adj = None
else:
    args.adj = adj

# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))

model = getattr(Models, args.model)(args)
if args.test or args.retrain or args.tune:
    model = torch.load(modelpath + '.pt')
    print('Loaded models from file.')
model.cuda()

# LOSS
criterion = getattr(torch.nn, args.loss)()

# OPTIM
optimizer = getattr(torch.optim, args.optim)(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)


# TRAINING
def train_model():
    loss_train = []
    size = inp_train.size(0)
    days = torch.randperm(size).cuda()
    iters = size // args.batch
    for day in range(iters):
        idx = days[day::iters]
        inp = inp_train[idx]
        tgt = tgt_train[idx]
        dt = dt_train[idx] if args.daytime else None
        out = model(inp, dt)
        if type(out) is tuple:
            out = out[0]
        out = Utils.denormalize(out, flow_mean, flow_std).contiguous()
        loss = criterion(out, tgt)
        loss.backward()
        clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()
        loss_train.append(loss.data[0])
    return sum(loss_train) / len(loss_train)


def valid_model():
    inp = inp_valid
    tgt = tgt_valid
    dt = dt_valid if args.daytime else None
    out = model(inp, dt)
    if type(out) is tuple:
        out = out[0]
    out = Utils.denormalize(out, flow_mean, flow_std)
    loss = criterion(out, tgt).data[0]
    return loss


def test_model():
    def percent_err(out, tgt):
        return float(criterion(out, tgt).data[0] / tgt.mean())

    inp = inp_test
    tgt = tgt_test
    dt = dt_test if args.daytime else None
    out = model(inp, dt)
    ret_more = False
    if type(out) is tuple:
        out, out_ = out[0], out[1:]
        ret_more = True
    out = Utils.denormalize(out, flow_mean, flow_std)
    loss = [percent_err(out[:, :, i], tgt[:, :, i])
            for i in range(args.future)]
    if ret_more:
        return loss, out_
    else:
        return loss


def run():
    if not args.test:
        for epoch in range(args.epoches):
            loss_train = train_model()
            loss_valid = valid_model()

            print('Epoch: %d train: %.4f valid: %.4f' % (
                epoch, loss_train, loss_valid))

            scheduler.step(loss_valid)
            if optimizer.param_groups[0]['lr'] < args.lr_min:
                break

    out = test_model()
    if type(out) is tuple:
        out, out_ = out
        Utils.torch2npsave(modelpath + '_out', out_)
    print('Test {}: {}'.format(modelpath, out))
    np.savetxt(modelpath + '_loss.txt', out)
    try:
        model.reset()
    except AttributeError:
        pass
    torch.save(model.cpu(), modelpath + '.pt')


if __name__ == '__main__':
    run()
