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
print(str(args))

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
data_train, data_valid, data_test, mean, std = pt_utils.load_dataset(args)
print(*[f'{key}:\t{len(data)}'
        for key, data in zip(['train', 'valid', 'test'],
                             [data_train, data_valid, data_test])], sep='\n')
data_train, data_valid, data_test = pt_utils.dataset_to_dataloader(
    data_train, data_valid, data_test, args.batch_size
)
adj = pt_utils.load_adj(args.dataset)

if args.cuda:
    mean, std, adj = mean.cuda(), std.cuda(), adj.cuda()

# MODEL
model = builder.build_model(args, adj)
if args.test or args.retrain:
    model.load_state_dict(torch.load(args.path + '.pt'))
if args.cuda:
    model.cuda()

# rescaler, criterion, metrics
rescaler = Trainer.Rescaler(mean, std)
criterion = getattr(torch.nn, args.criterion)()
loss = Loss.Loss(args.metrics, args.futures)

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

error, info = trainer.eval(data_test)
print(f'Test {args.path}:\n{error}')
torch.save(model.state_dict(), args.path + '.pt')
if info:
    pickle.dump(info, open(args.path + '.pkl', 'wb'))
