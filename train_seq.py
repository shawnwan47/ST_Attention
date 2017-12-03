import argparse

import torch
import torch.nn as nn

import Models
import Optim
import Loss
import Utils

parser = argparse.ArgumentParser(
    description='Seq2Seq',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('-data_type', type=str, default='seq')
parser.add_argument('-gran', type=int, default=15)
parser.add_argument('-past', type=int, default=8)
parser.add_argument('-future', type=int, default=4)
# model
parser.add_argument('-rnn_type', type=str, default='GRU',
                    choices=['GRU', 'LSTM', 'RNN'])
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
parser.add_argument('-lr_patience', type=int, default=5)
parser.add_argument('-weight_decay', type=float, default=1e-5)
parser.add_argument('-max_grad_norm', type=float, default=1.)
parser.add_argument('-beta1', type=float, default=0.9)
parser.add_argument('-beta2', type=float, default=0.98)
# gpu
parser.add_argument('-gpuid', default=[], nargs='+', type=int)
parser.add_argument('-seed', type=int, default=47)
# file
parser.add_argument('-savepath', type=str, default='save.pt')

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
inputs_valid.volatile = True
targets_valid.volatile = True
targets_valid_raw = targets_valid * flow_std + flow_mean

# MODEL
ndim = inputs_train.size(-1)
model = Models.Flow2Flow(args)
model = model.cuda() if args.gpuid else model

# optimizer
criterion = nn.MSELoss()
optimizer = Optim.Optim(args)
optimizer.set_parameters(model.parameters())

# training
t = args.past
for epoch in range(args.nepoch):
    # train
    loss_train = []
    for d in torch.randperm(inputs_train.size(1)):  # random day
        src = inputs_train[:t, d]
        tgt = inputs_train[t:, d]
        outputs = model(src, tgt, teach=True)
        loss = criterion(outputs[0], targets_train[t:])
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data[0])
    loss_train = sum(loss_train) / len(loss_train)

    # valid
    src = inputs_valid[:t]
    tgt = inputs_valid[t:]
    outputs = model(src, tgt, teach=False)
    loss_valid = criterion(outputs[0], targets_valid[t:])
    outputs = outputs[0] * flow_std + flow_mean
    wape_valid = Loss.WAPE(outputs, targets_valid_raw[t:])
    mape_valid = Loss.MAPE(outputs, targets_valid_raw[t:])

    # update
    print('epoch: %d train: %.4f valid: %.4f WAPE: %.4f MAPE: %.4f' % (
        epoch, loss_train, loss_valid, wape_valid, mape_valid))

    optimizer.updateLearningRate(loss_valid)

torch.save(model, args.savepath)
