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


args = argparse.ArgumentParser()
config.add_data(args)
config.add_model(args)
config.add_train(args)
args = args.parse_args()
config.update_data(args)
config.update_model(args)

# CUDA
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpuid)
    float_type = torch.cuda.FloatTensor
    long_type = torch.cuda.LongTensor
    torch.cuda.manual_seed(args.seed)
    print(f'Using GPU: {args.gpuid}')
else:
    torch.manual_seed(args.seed)
    float_type = torch.FloatTensor
    long_type = torch.LongTensor
    print('Using CPU')

# DATA
data_train, data_valid, data_test, mean, std = pt_utils.load_dataset(
    dataset=args.dataset,
    freq=args.freq,
    history=args.history,
    horizon=args.horizon,
    batch_size=args.batch_size
)

if args.cuda:
    mean, std = mean.cuda(), std.cuda()

# MODEL
model = builder.build_model(args)
if args.test or args.retrain:
    model.load_state_dict(torch.load(args.path + '.pt'))
if args.cuda:
    model.cuda()
print(f'{args.path} parameters: {pt_utils.count_parameters(model)}')

# rescaler, criterion, loss
rescaler = Trainer.Rescaler(mean, std)
criterion = getattr(torch.nn, args.criterion)()
loss = Loss.Loss(metrics=args.metrics, horizons=args.horizons)

# optimizer, scheduler
if args.optim is 'SGD':
    optimizer = optim.SGD(model.parameters(),
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=True)
else:
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoches)

# TRAINER
trainer = Trainer.Trainer(model, rescaler, criterion, loss,
                          optimizer, scheduler, args.epoches, args.cuda)


if not args.test:
    trainer.run(data_train, data_valid, data_test)

error = trainer.eval(data_test)
print(f'Test {args.path}:\n{error}')
torch.save(trainer.model.state_dict(), args.path + '.pt')
# if info:
#     pickle.dump(info, open(args.path + '.pkl', 'wb'))
