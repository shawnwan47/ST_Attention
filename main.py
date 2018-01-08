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

# DATA
(inp_train, inp_valid, inp_test,
 tgt_train, tgt_valid, tgt_test,
 st_train, st_valid, st_test,
 flow_min, flow_scale) = Utils.load_data(args)


# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))

model = getattr(Models, args.model)(args)
if args.test:
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
def train():
    model.train()
    loss_train = []
    num_sample = inp_train.size(0)
    days = torch.randperm(num_sample)
    iters = num_sample // args.batch
    for day in range(iters):
        idx = days[day::iters]
        inp = inp_train[idx].cuda()
        tgt = tgt_train[idx].cuda()
        st = st_train[idx].cuda()
        out = model(inp, st)
        if type(out) is tuple:
            out = out[0]
        print(out[0])
        loss = criterion(out, tgt.view(-1))
        loss_train.append(loss.data[0])
        # optimization
        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return sum(loss_train) / len(loss_train)


def valid():
    model.eval()
    loss = 0
    iters = inp_valid.size(0) // args.batch
    for i in range(iters):
        inp = inp_valid[i::iters].cuda()
        tgt = tgt_valid[i::iters].cuda()
        st = st_valid[i::iters].cuda()
        out = model(inp, st)
        if type(out) is tuple:
            out = out[0]
        loss += criterion(out, tgt.view(-1)).data[0]
    return loss / iters


def test():
    model.eval()
    loss = 0
    iters = inp_test.size(0) // args.batch
    for i in range(iters):
        inp = inp_test[i::iters].cuda()
        tgt = tgt_test[i::iters].cuda()
        st = st_test[i::iters].cuda()
        out = model(inp, st)
        ret_more = False
        if type(out) is tuple:
            out, out_ = out[0], out[1:]
            ret_more = True
        loss += criterion(out, tgt.view(-1)).data[0]
    loss = loss / iters
    if ret_more:
        return loss, out_
    else:
        return loss


def run():
    if not args.test:
        for epoch in range(args.epoches):
            loss_train = train()
            loss_valid = valid()

            if not epoch % args.print_epoches:
                print('Epoch: %d train: %.4f valid: %.4f' % (
                    epoch, loss_train, loss_valid))

            scheduler.step(loss_valid)
            if optimizer.param_groups[0]['lr'] < args.lr_min:
                break

    out = test()
    if type(out) is tuple:
        out, out_ = out[0], out[1]
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
