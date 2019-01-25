import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from config import Config
from lib import TimeSeriesLoss, MetricDict, mask_target
from lib.io import load_data
from builder import build_model


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


def train(**kwargs):
    config = Config(**kwargs)
    _cuda(config)
    loader_train, loader_valid, loader_test, mean, std = load_data(config)
    model = build_model(config, mean, std)
    criterion = getattr(nn, config.criterion)()
    loss = TimeSeriesLoss(metrics=config.metrics, horizons=config.horizons)
    optimizer = Adam(model.parameters(), weight_decay=config.weight_decay)

    crit_best = float('inf')
    trial = 0
    for epoch in range(config.epoches):
        model.train()
        loss_train = MetricDict()
        for data in loader_train:
            if config.cuda:
                data = (d.cuda() for d in data)
            data, time, weekday, target = data
            output = model(data, time, weekday)
            loss_train += loss(output, target)
            output, target = mask_target(output, target)
            crit = criterion(output, target)
            optimizer.zero_grad()
            crit.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            del output, crit
        loss_valid = _eval(model, loader_valid, loss, config.cuda)
        print(f'Epoch: {epoch}\nTrain: {loss_train}\nValid: {loss_valid}')
        # early stopping
        crit_avg = loss.get_criterion(loss_valid)
        if crit_avg < crit_best:
            crit_best = crit_avg
            trial = 0
            torch.save(model.state_dict(), config.path)
        else:
            trial += 1
        if trial > config.patience:
            break
    model.load_state_dict(torch.load(config.path))
    loss_test = _eval(model, loader_test, loss, config.cuda)
    print(f'Test:{loss_test}')


def test(**kwargs):
    config = Config(**kwargs)
    _cuda(config)
    _, _, dataloader, mean, std = load_data(config)
    model = build_model(config, mean, std)
    model.load_state_dict(torch.load(config.path))
    loss = TimeSeriesLoss(metrics=config.metrics, horizons=config.horizons)
    loss_test = _eval(model, dataloader, loss, config.cuda)
    print(error)


def _eval(model, dataloader, loss, cuda):
    model.eval()
    loss_total = MetricDict()
    error = 0
    for data in dataloader:
        if cuda:
            data = (d.cuda() for d in data)
        data, time, weekday, target = data
        output = model(data, time, weekday)
        loss_total += loss(output, target)
    return loss_total


if __name__ == '__main__':
    import fire
    fire.Fire()
