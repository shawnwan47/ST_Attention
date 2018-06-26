import argparse
import pickle

import numpy as np

import torch
import torch.optim as optim

from lib import config
from lib import pt_utils
from lib import Loss
from lib import Interpreter

from models import builder


args = argparse.ArgumentParser()
config.add_data(args)
config.add_model(args)
config.add_device(args)
config.add_train(args)
args = args.parse_args()
config.update_data(args)
config.update_model(args)

# CUDA
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpuid)
    torch.cuda.manual_seed(args.seed)
    print(f'Using GPU: {args.gpuid}')
else:
    torch.manual_seed(args.seed)
    print('Using CPU')

# DATA
data_loaders, mean, std = pt_utils.load_dataset(
    dataset=args.dataset,
    freq=args.freq,
    history=args.history,
    horizon=args.horizon,
    batch_size=args.batch_size
)
_, _, _, data_case = data_loaders

if args.cuda:
    mean, std = mean.cuda(), std.cuda()

# MODEL
model = builder.build_model(args)
model.load_state_dict(torch.load(args.path + '.pt'))
if args.cuda:
    model.cuda()
print(f'{args.path} parameters: {pt_utils.count_parameters(model)}')

# rescaler, loss
rescaler = pt_utils.Rescaler(mean, std)
loss = Loss.Loss(metrics=args.metrics, horizons=args.horizons)

# interpreter
interpreter = Interpreter.Interpreter(
    model=model,
    rescaler=rescaler,
    loss=loss,
    cuda=args.cuda
)

targets, outputs, infos = interpreter.eval(data_case)
for i, (target, output) in enumerate(zip(targets, outputs)):
    target.dump(args.path + '_target_' + str(i))
    output.dump(args.path + '_output_' + str(i))
for i, info_i in enumerate(infos):
    for j, info_j in enumerate(info_i):
        info_j.dump(args.path + '_info_' + str(i) + str(j))
