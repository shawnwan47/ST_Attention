import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

import Models
import Optim
import Loss
import Utils

parser = argparse.ArgumentParser(
    description='Seq2Seq',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('-data_type', type=str, default='seq')
parser.add_argument('-start_time', type=int, default=6)
parser.add_argument('-gran', type=int, default=15)
parser.add_argument('-past', type=int, default=8)
parser.add_argument('-future', type=int, default=4)
# model
parser.add_argument('-rnn_type', type=str, default='GRU',
                    choices=['GRU', 'LSTM', 'RNN'])
parser.add_argument('-ndim', type=int, default=538)
parser.add_argument('-nhid', type=int, default=512)
parser.add_argument('-nlay', type=int, default=1)
parser.add_argument('-pdrop', type=float, default=0.1)
parser.add_argument('-bidirectional', action='store_true')
parser.add_argument('-attention', action='store_true')
parser.add_argument('-attention_type', type=str, default='mlp',
                    choices=['dot', 'general', 'mlp'])
# train
parser.add_argument('-nepoch', type=int, default=100)
parser.add_argument('-niter', type=int, default=10)
# optimizer
parser.add_argument('-optim_method', type=str, default='sgd',
                    choices=['sgd', 'adagrad', 'adadelta', 'adam'])
parser.add_argument('-lr', type=float, default=0.1)
parser.add_argument('-lr_min', type=float, default=1e-8)
parser.add_argument('-lr_decay', type=float, default=0.1)
parser.add_argument('-patience', type=int, default=5)
parser.add_argument('-weight_decay', type=float, default=1e-5)
parser.add_argument('-max_grad_norm', type=float, default=1.)
parser.add_argument('-beta1', type=float, default=0.9)
parser.add_argument('-beta2', type=float, default=0.98)
# gpu
parser.add_argument('-gpuid', default=[], nargs='+', type=int)
parser.add_argument('-seed', type=int, default=47)

args = parser.parse_args()
print(args)

if torch.cuda.is_available() and not args.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

if args.attention:
    print('Training RNN with Attention')
else:
    print('Training RNN without Attention')

# DATA
(inputs_train, inputs_valid, _,
 targets_train, targets_valid, _,
 daytimes_train, daytimes_valid, _,
 flow_mean, flow_std) = Utils.load_data(args)
inputs_train = Variable(inputs_train)
targets_train = Variable(targets_train)
inputs_valid = Variable(inputs_valid, volatile=True)
targets_valid = Variable(targets_valid, volatile=True)
flow_mean = Variable(flow_mean, requires_grad=False)
flow_std = Variable(flow_std, requires_grad=False)
targets_valid_raw = targets_valid * flow_std + flow_mean

# MODEL
model = Models.Seq2Seq(args)
model = model.cuda() if args.gpuid else model

# optimizer
criterion = nn.MSELoss()
optimizer = Optim.Optim(args)
optimizer.set_parameters(model.parameters())

# training
for epoch in range(args.nepoch):
    # train for random day
    loss_train = []
    for d in torch.randperm(inputs_train.size(1)):
        src = inputs_train[:args.past, d].unsqueeze(1)
        tgt = inputs_train[args.past:, d].unsqueeze(1)
        outputs = model(src, tgt, teach=True)
        loss = criterion(outputs[0], targets_train[args.past:, d])
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data[0])
    loss_train = sum(loss_train) / len(loss_train)

    # valid for every time
    loss_valid = []
    wape_valid = []
    mape_valid = []
    for t in range(args.past, inputs_valid.size(0) - args.future):
        src = inputs_valid[:t]
        tgt = inputs_valid[t:t + args.future]
        tgt_out = targets_valid[t:t + args.future]
        tgt_out_raw = targets_valid_raw[t:t + args.future]
        outputs = model(src, tgt, teach=False)
        loss_valid.append(criterion(outputs[0], tgt_out).data[0])
        outputs = outputs[0] * flow_std + flow_mean
        wape_valid.append(Loss.WAPE(outputs, tgt_out_raw).data[0])
        mape_valid.append(Loss.MAPE(outputs, tgt_out_raw).data[0])
    loss_valid = sum(loss_valid) / len(loss_valid)
    wape_valid = sum(wape_valid) / len(wape_valid)
    mape_valid = sum(mape_valid) / len(mape_valid)

    # update
    print('epoch: %d train: %.4f valid: %.4f WAPE: %.4f MAPE: %.4f' % (
        epoch, loss_train, loss_valid, wape_valid, mape_valid))

    optimizer.updateLearningRate(loss_valid)

torch.save(model, Utils.modelname(args))
