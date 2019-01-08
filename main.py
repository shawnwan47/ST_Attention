import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import Config
from lib.metric import Loss, MetricDict
from lib.io import load_data
from builder import build_model


def train(**kwargs):
    print(kwargs)
    config = Config(**kwargs)
    _cuda(config)
    loader_train, loader_validation, loader_test, mean, std = load_data(config)
    model = build_model(config, mean, std)
    criterion = getattr(nn, config.criterion)()
    loss = Loss(metrics=config.metrics, horizons=config.horizons)
    optimizer = Adam(model.parameters(), weight_decay=config.weight_decay)

    for epoch in range(config.epoches):
        model.train()
        error = MetricDict()
        for data in loader_train:
            print(len(data))
            if config.cuda:
                data = (d.cuda() for d in data)
            data, time, weekday, target = data
            output = model(data, time, weekday)
            error = error + loss(output, target)
            criterion = criterion(output, target)
            optimizer.zero_grad()
            criterion.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        error_validation = _eval(model, loader_validation, loss, config.cuda)
        print(error, error_validation)
        torch.save(model.state_dict(), config.path)
    error_test = _eval(model, loader_test, loss, config.cuda)
    print(error_test)


def test(**kwargs):
    config = Config(**kwargs)
    _cuda(config)
    _, _, dataloader, mean, std = load_data(config)
    model = build_model(config, mean, std)
    model.load_state_dict(torch.load(config.path))
    loss = Loss(metrics=config.metrics, horizons=config.horizons)
    error = _eval(model, dataloader, loss, config.cuda)
    print(error)


def _cuda(config):
    config.cuda &= torch.cuda.is_available()
    if config.cuda:
        torch.cuda.set_device(config.gpuid)
        print(f'Using GPU: {config.gpuid}')
    else:
        print('Using CPU')
    if config.seed is not None:
        if config.cuda:
            torch.cuda.manual_seed(config.seed)
        else:
            torch.manual_seed(config.seed)


def _eval(model, dataloader, loss, cuda):
    model.eval()
    error = MetricDict()
    for data in dataloader:
        if cuda:
            data = (d.cuda() for d in data)
        data, time, weekday, target = data
        output = model(data, time, weekday)
        error = error + loss(output, target)
    return error


if __name__ == '__main__':
    import fire
    fire.Fire()
