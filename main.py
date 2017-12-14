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
 daytime_train, daytime_valid, daytime_test,
 data_mean, data_std) = data


inp_train = Variable(inp_train)
inp_valid = Variable(inp_valid, volatile=True)
inp_test = Variable(inp_test, volatile=True)
daytime_train = Variable(daytime_train)
daytime_valid = Variable(daytime_valid, volatile=True)
daytime_test = Variable(daytime_test, volatile=True)
data_mean = Variable(data_mean, requires_grad=False)
data_std = Variable(data_std, requires_grad=False)

# MODEL
modelpath = MODEL_PATH + Args.modelname(args)
print('Model: {}'.format(modelpath))
model = getattr(Models, args.model)(args).cuda()
emb_dt = Models.DayTime(args).cuda()

# LOSS
criterion = getattr(torch.nn, args.loss)()

# OPTIM
parameters = list(model.parameters()) + list(emb_dt.parameters())
optimizer = getattr(torch.optim, args.optim)(
    parameters, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.patience, verbose=True)


def train_model():
    loss_train = []
    for d in torch.randperm(inp_train.size(0)):
        inp = inp_train[d]
        tgt = tgt_train[d]
        if args.daytime:
            inp = torch.cat((inp, emb_dt(daytime_train[d])), -1)
        out = model(inp)
        out = Utils.denormalize(out[0], data_mean, data_std)
        loss = criterion(out, tgt)
        loss.backward()
        clip_grad_norm(parameters, args.max_grad_norm)
        optimizer.step()
        loss_train.append(loss.data[0])
    return sum(loss_train) / len(loss_train)


def valid_model():
    inp = inp_valid
    out = model(inp)
    out = Utils.denormalize(out[0])
    loss = criterion(out, tgt_valid).data[0]
    return loss


def eval_model():
    def percent_err(out, tgt):
        return criterion(out, tgt).data[0] / tgt.mean()

    def nth_future(data, i):
        return data[:, :, i * args.dim:(i + 1) * args.dim]

    out, attn = model(inp_test)
    out = Utils.denormalize(out, data_mean, data_std)
    loss = [percent_err(nth_future(out, i), nth_future(tgt_test, i))
            for i in range(args.future)]
    return np.array(list(map(float, loss))), attn


# TRAINING
if not args.test:
    for epoch in range(args.epoches):
        loss_train = train_model()
        loss_valid = valid_model()

        print('Epoch: %d train: %.4f valid: %.4f' % (
            epoch, loss_train, loss_valid))

        scheduler.step(loss_valid)

    torch.save(model.cpu(), modelpath + '.pt')

# TESTING
model = torch.load(modelpath + '.pt').cuda()

loss_test, attn = eval_model()
print('Test {}: {}'.format(modelpath, loss_test))
np.savetxt(modelpath + '_loss.txt', loss_test)

if (args.model == 'RNN' and args.attn) or args.model == 'Transformer':
    np.save(modelpath + '_attn', attn)
