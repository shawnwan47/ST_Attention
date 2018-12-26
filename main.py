import numpy as np

import torch
import torch.optim as optim
import fire

from config import Args
from lib import utils
from lib import pt_utils
from lib import Loss
from lib import Trainer

from models import builder


args = Args()
loader_train, loader_valid, loader_test, mean, std = pt_utils.load_loaders(args)

model = builder.build(args, mean, std)
print(f'{args.path} parameters: {pt_utils.count_parameters(model)}')
if args.cuda:
    model.cuda()

criterion = getattr(torch.nn, args.criterion)()
loss = Loss.Loss(metrics=args.metrics, horizons=args.horizons)
optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

def test():
    model.load_state_dict(torch.load(args.path))
    model.eval()

def train( dataloader):
    error = Loss.MetricDict()
    for data in dataloader:
        if args.cuda:
            data = (d.cuda() for d in data)
        output = model(*data)
        error = error + loss(output, target)
        criterion = criterion(output, target)
        optimizer.zero_grad()
        criterion.backward()
        clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return error


torch.save(trainer.model.state_dict(), args.path + '.pt')

error_test = trainer.run_epoch(loader_test)
print(f'{args.path}:\n{error_test}')


if __name__ == '__main__':
    fire.Fire()
