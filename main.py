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
    torch.cuda.set_device(args.gpuid[0])
if args.seed > 0:
    torch.cuda.manual_seed(args.seed)

# DATA
(flow_train, flow_valid, flow_test,
 diff_train, diff_valid, diff_test,
 tgt_train, tgt_valid, tgt_test,
 daytime_train, daytime_valid, daytime_test,
 flow_mean, flow_std) = getattr(Utils, 'load_data_' + args.data_type)(args)

diff_train = Variable(diff_train)
diff_valid = Variable(diff_valid, volatile=True)
diff_test = Variable(diff_test, volatile=True)
flow_train = Variable(flow_train)
flow_valid = Variable(flow_valid, volatile=True)
flow_test = Variable(flow_test, volatile=True)
flow_mean = Variable(flow_mean)
flow_std = Variable(flow_std)
tgt_train = Variable(tgt_train)
tgt_valid = Variable(tgt_valid, volatile=True)
tgt_test = Variable(tgt_test, volatile=True)
daytime_train = Variable(daytime_train)
daytime_valid = Variable(daytime_valid, volatile=True)
daytime_test = Variable(daytime_test, volatile=True)

def recover_flow(diff, flow):
    flow_cum = flow[:, args.past:].unsqueeze(-2) + diff.cumsum(-2)
    return flow_cum * flow_std + flow_mean


# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))

model = getattr(Models, args.model)(args)
if args.test or args.retrain:
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
    model.train()
    loss_train = []
    size = diff_train.size(0)
    days = torch.randperm(size).cuda()
    iters = size // args.batch
    for day in range(iters):
        idx = days[day::iters]
        flow = flow_train[idx]
        diff = diff_train[idx]
        tgt = tgt_train[idx]
        daytime = daytime_train[idx] if args.daytime else None
        out = model(diff, daytime)
        if type(out) is tuple:
            out, out_ = out[0], out[1:]
        out = recover_flow(out, flow)
        loss = criterion(out, tgt)
        loss_train.append(loss.data[0])
        if type(out) is tuple and args.reg:
            loss += args.reg_weight * model.regularizer(out_)
        loss.backward()
        clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return sum(loss_train) / len(loss_train)


def valid_model():
    model.eval()
    flow = flow_valid
    diff = diff_valid
    tgt = tgt_valid
    daytime = daytime_valid if args.daytime else None
    out = model(diff, daytime)
    if type(out) is tuple:
        out = out[0]
    out = recover_flow(out, flow)
    loss = criterion(out, tgt).data[0]
    return loss


def test_model():
    def percent_err(out, tgt):
        return float(criterion(out, tgt).data[0] / tgt.mean())

    model.eval()
    flow = flow_test
    diff = diff_test
    tgt = tgt_test
    daytime = daytime_test if args.daytime else None
    out = model(diff, daytime)
    ret_more = False
    if type(out) is tuple:
        out, out_ = out[0], out[1:]
        ret_more = True
    out = recover_flow(out, flow)
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

            if not epoch % args.print_epoches:
                print('Epoch: %d train: %.4f valid: %.4f' % (
                    epoch, loss_train, loss_valid))

            scheduler.step(loss_valid)
            if optimizer.param_groups[0]['lr'] < args.lr_min:
                break

    out = test_model()
    if type(out) is tuple:
        out, out_ = out[0], out[1:]
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
