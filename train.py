import argparse
import pickle

import numpy as np

import torch
import torch.optim as optim

from models import Models
import utils


args = argparse.ArgumentParser()
utils.config.add_data(args)
utils.config.add_model(args)
utils.config.add_train(args)
args = args.parse_args()
utils.config.update(args)
print(args)

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
data_train, data_valid, data_test, mean, scale, adj = utils.pt.get_dataset(
    dataset=args.dataset,
    freq=args.freq,
    start=args.start,
    end=args.end,
    batch_size=args.batch_size,
    cuda=args.cuda
)

# MODEL
model = model.Models.build_model(args)
if args.test or args.retrain:
    model.load_state_dict(torch.load(args.path + '.pt'))
if args.cuda:
    model.cuda()

# LOSS & OPTIM
loss = utils.loss.Loss(args.loss, mean, scale)

if args.optim is 'SGD':
    optimizer = optim.SGD(model.parameters(),
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=True)
else:
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

# TRAINER
trainer = utils.trainer.Trainer(model, loss, optimizer, scheduler,
                                args.max_grad_norm, args.cuda)

if not args.test:
    for epoch in range(args.epoches):
        trainer.run_epoch(data_train, data_valid, data_test)
        if trainer.optimizer.param_groups[0]['lr'] < args.min_lr:
            break

error, info = trainer.eval(data_test)
print(f'Test\t{error}')
torch.save(model.state_dict(), args.path + '.pt')
if info:
    pickle.dump(info, open(args.path + '.pkl', 'wb'))
