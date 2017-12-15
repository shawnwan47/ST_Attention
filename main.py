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
 flow_mean, flow_std) = data


inp_train = Variable(inp_train)
inp_valid = Variable(inp_valid, volatile=True)
inp_test = Variable(inp_test, volatile=True)
dt_train = Variable(dt_train)
dt_valid = Variable(dt_valid, volatile=True)
dt_test = Variable(dt_test, volatile=True)
tgt_train = Variable(tgt_train)
tgt_valid = Variable(tgt_valid, volatile=True)
tgt_test = Variable(tgt_test, volatile=True)
flow_mean = Variable(flow_mean, requires_grad=False)
flow_std = Variable(flow_std, requires_grad=False)

# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))
model = getattr(Models, args.model)(args).cuda()
daytime = Models.Embedding_DayTime(args).cuda()

# LOSS
criterion = getattr(torch.nn, args.loss)()

# OPTIM
parameters = list(model.parameters()) + list(daytime.parameters())
optimizer = getattr(torch.optim, args.optim)(
    parameters, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.patience, verbose=True)


# TRAINING
def train_model():
    loss_train = []
    for d in torch.randperm(inp_train.size(0)):
        inp = inp_train[d].unsqueeze(0)
        tgt = tgt_train[d].unsqueeze(0)
        dt = dt_train[d].unsqueeze(0)
        if args.daytime:
            inp = torch.cat((inp, daytime(dt)), -1)
        out = model(inp)
        if type(out) is tuple:
            out, attn = out
        out = Utils.denormalize(out, flow_mean, flow_std).contiguous()
        loss = criterion(out, tgt)
        loss.backward()
        clip_grad_norm(parameters, args.max_grad_norm)
        optimizer.step()
        loss_train.append(loss.data[0])
    return sum(loss_train) / len(loss_train)


def valid_model():
    inp = inp_valid
    if args.daytime:
        inp = torch.cat((inp, daytime(dt_valid)), -1)
    out = model(inp)
    if type(out) is tuple:
        out, attn = out
    out = Utils.denormalize(out, flow_mean, flow_std)
    loss = criterion(out, tgt_valid).data[0]
    return loss


def test_model():
    def percent_err(out, tgt):
        return float(criterion(out, tgt).data[0] / tgt.mean())

    inp = inp_test
    if args.daytime:
        inp = torch.cat((inp, daytime(dt_test)), -1)
    out = model(inp)
    ret_attn = False
    if type(out) is tuple:
        out, attn = out
        ret_attn = True
    out = Utils.denormalize(out, flow_mean, flow_std)
    loss = np.array([percent_err(out[:, :, i], tgt_test[:, :, i])
                     for i in range(args.future)])
    return (loss, attn) if ret_attn else loss


# TRAINING
if not args.test:
    for epoch in range(args.epoches):
        loss_train = train_model()
        loss_valid = valid_model()

        print('Epoch: %d train: %.4f valid: %.4f' % (
            epoch, loss_train, loss_valid))

        scheduler.step(loss_valid)

    torch.save((model.cpu(), daytime.cpu()), modelpath + '.pt')

# TESTING
model, daytime = torch.load(modelpath + '.pt')
model = model.cuda()
daytime = daytime.cuda()

out = test_model()
if type(out) is tuple:
    out, attn = out
    np.save(modelpath + '_attn', attn.cpu().data.numpy())
print('Test {}: {}'.format(modelpath, out))
np.savetxt(modelpath + '_loss.txt', out)
