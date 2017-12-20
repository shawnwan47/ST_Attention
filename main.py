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

# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))

model = getattr(Models, args.model)(args)
daytime = Models.Embedding_DayTime(args)
if args.test or args.retrain or args.tune:
    model, daytime = torch.load(modelpath + '.pt')
    print('Loaded models from file.')
if args.fix_layers:
    model.fix_layers(args.fix_layers)
    daytime.fix()
if args.eval_layers:
    model.set_eval_layers(args.eval_layers)
model.cuda()
daytime.cuda()

# LOSS
criterion = getattr(torch.nn, args.loss)()

# OPTIM
parameters = list(model.parameters()) + list(daytime.parameters())
optimizer = getattr(torch.optim, args.optim)(
    parameters, lr=args.lr, weight_decay=args.weight_decay)
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
        dt = dt_train[idx]
        tgt = tgt_train[idx]
        if args.daytime:
            inp = torch.cat((inp, daytime(dt)), -1)
        out = model(inp)
        if type(out) is tuple:
            out = out[0]
        out = Utils.denormalize(out, flow_mean, flow_std).contiguous()
        loss = criterion(out, tgt)
        loss.backward()
        clip_grad_norm(parameters, args.max_grad_norm)
        optimizer.step()
        loss_train.append(loss.data[0])
    return sum(loss_train) / len(loss_train)


def valid_model():
    inp = inp_valid
    tgt = tgt_valid
    dt = dt_valid
    if args.daytime:
        inp = torch.cat((inp, daytime(dt)), -1)
    out = model(inp)
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
    dt = dt_test
    if args.daytime:
        inp = torch.cat((inp, daytime(dt)), -1)
    out = model(inp)
    ret_attn = False
    if type(out) is tuple:
        out, attn = out[0], out[1:]
        ret_attn = True
    out = Utils.denormalize(out, flow_mean, flow_std)
    loss = [percent_err(out[:, :, i], tgt[:, :, i])
            for i in range(args.future)]
    if ret_attn:
        return loss, attn
    else:
        return loss


def run():
    # TRAINING
    if not args.test:
        for epoch in range(args.epoches):
            loss_train = train_model()
            loss_valid = valid_model()

            print('Epoch: %d train: %.4f valid: %.4f' % (
                epoch, loss_train, loss_valid))

            scheduler.step(loss_valid)
            if optimizer.param_groups[0]['lr'] <= args.lr_min:
                break

    # TESTING
    model.cuda()
    daytime.cuda()
    out = test_model()
    if type(out) is tuple:
        out, attn = out
        Utils.torch2npsave(modelpath + '_attn', attn)
    print('Test {}: {}'.format(modelpath, out))
    np.savetxt(modelpath + '_loss.txt', out)
    if args.model in ['LinearSpatial', 'LinearTemporal',
                      'LinearSpatialTemporal', 'LinearST']:
        Utils.torch2npsave(modelpath + '_params', list(model.parameters()))

    try:
        model.reset()
    except AttributeError:
        pass
    daytime.reset()
    torch.save((model.cpu(), daytime.cpu()), modelpath + '.pt')


if __name__ == '__main__':
    run()
