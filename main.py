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
data_train, data_validation, data_test, mean, std = pt_utils.load_data(args)
model = builder.build_model(args, mean, std)
criterion = getattr(torch.nn, args.criterion)()
loss = Loss.Loss(metrics=args.metrics, horizons=args.horizons)
optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)


def test(dataloader):
    model.load_state_dict(torch.load(args.path))
    model.eval()
    error = Loss.MetricDict()
    for data in dataloader:
        if args.cuda:
            data = (d.cuda() for d in data)
        output = model(*data)
        error = error + loss(output, target)
    return error


def train():
    error = Loss.MetricDict()
    for epoch in args.epoches:
        for data in loader_train:
            if args.cuda:
                data = (d.cuda() for d in data)
            output = model(*data)
            error = error + loss(output, target)
            criterion = criterion(output, target)
            optimizer.zero_grad()
            criterion.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

    torch.save(model.state_dict(), args.path)
    return error


if __name__ == '__main__':
    fire.Fire()
