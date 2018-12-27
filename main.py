import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import Config
from lib.Loss import Loss

from models import builder


config = Config()
data_train, data_validation, data_test, mean, std = pt_utils.load_data(config)
model = builder.build_model(config, mean, std)
criterion = getattr(nn, config.criterion)()
loss = Loss(metrics=config.metrics, horizons=config.horizons)
optimizer = Adam(model.parameters(), weight_decay=config.weight_decay)


def test(dataloader):
    model.load_state_dict(torch.load(config.path))
    model.eval()
    error = Loss.MetricDict()
    for data in dataloader:
        if config.cuda:
            data = (d.cuda() for d in data)
        output = model(*data)
        error = error + loss(output, target)
    return error


def train():
    error = Loss.MetricDict()
    for epoch in config.epoches:
        for data in loader_train:
            if config.cuda:
                data = (d.cuda() for d in data)
            output = model(*data)
            error = error + loss(output, target)
            criterion = criterion(output, target)
            optimizer.zero_grad()
            criterion.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

    torch.save(model.state_dict(), config.path)
    return error


if __name__ == '__main__':
    import fire
    fire.Fire()
