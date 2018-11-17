import argparse
import pickle

import numpy as np

import torch
import torch.optim as optim

from lib import config
from lib import pt_utils
from lib import Loss
from lib import Trainer

from models import builder

# run with "python train.py -cuda -model=GARNN"


args = argparse.ArgumentParser()
config.add_data(args)
config.add_device(args)
config.add_model(args)
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
<<<<<<< HEAD
data_loaders, mean, std = pt_utils.load_dataloaders(
=======
dataloader_train, dataloader_valid, dataloader_test, mean, std = pt_utils.load_data(
>>>>>>> b2a99cf93dd7a5a9f4edc617b2042d28cb22a0dc
    dataset=args.dataset,
    freq=args.freq,
    history=args.history,
    horizon=args.horizon,
    batch_size=args.batch_size
)
<<<<<<< HEAD
data_train, data_valid, data_test = data_loaders
=======
>>>>>>> b2a99cf93dd7a5a9f4edc617b2042d28cb22a0dc
if args.cuda:
    mean, std = mean.cuda(), std.cuda()
rescaler = pt_utils.Rescaler(mean, std)


# MODEL
model = builder.build_model(args)
if args.test:
    model.load_state_dict(torch.load(args.path + '.pt'))
if args.cuda:
    model.cuda()
print(f'{args.path} parameters: {pt_utils.count_parameters(model)}')

# criterion, loss
criterion = getattr(torch.nn, args.criterion)()
loss = Loss.Loss(metrics=args.metrics, horizons=args.horizons)

# optimizer, scheduler
optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

# TRAINER
trainer = Trainer.Trainer(
    model=model,
    rescaler=rescaler,
    criterion=criterion,
    loss=loss,
    optimizer=optimizer,
    epoches=args.epoches,
    iterations=args.iterations,
    cuda=args.cuda
)


if not args.test:
    trainer.run(dataloader_train, dataloader_valid)
    torch.save(trainer.model.state_dict(), args.path + '.pt')

error_test = trainer.run_epoch(dataloader_test)
print(f'{args.path}:\n{error_test}')
