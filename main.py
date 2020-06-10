import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config import Config
from lib import TimeSeriesLoss, MetricDict, mask_target
from lib.io import load_dataset
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
    config.set_dataset()
    config.set_model()
    _cuda(config)
    dataset_train, dataset_valid, dataset_test, mean, std, coordinates = load_dataset(config)
    loader_train = DataLoader(dataset_train, config.batch_size, True)
    loader_valid = DataLoader(dataset_valid, config.batch_size)
    loader_test = DataLoader(dataset_test, config.batch_size)
    model = build_model(config, mean, std, coordinates)
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
            output = output[0] if isinstance(output, tuple) else output
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
            torch.save(model.state_dict(), config.path_model)
        else:
            trial += 1
        if trial > config.patience:
            # early stopping
            break

    model.load_state_dict(torch.load(config.path_model))
    loss_test = _eval(model, loader_test, loss, config.cuda)
    print(f'{config.path_model}:\n{loss_test}')


def test(**kwargs):
    config = Config(**kwargs)
    config.set_dataset()
    config.set_model()
    _cuda(config)
    _, _, dataset_test, mean, std, coordinates = load_dataset(config)
    dataloader = DataLoader(dataset_test, config.batch_size)
    model = build_model(config, mean, std, coordinates)
    model.load_state_dict(torch.load(config.path_model))
    loss = TimeSeriesLoss(metrics=config.metrics, horizons=config.horizons)
    loss_test, output, attn = _eval(model, dataloader, loss, config.cuda, verbose=True)
    print(loss_test)
    output, attn = output[:config.num_times], attn[:config.num_times]
    np.save(open(config.path_result.with_suffix('.output'), 'wb'), output)
    np.save(open(config.path_result.with_suffix('.attn'), 'wb'), attn)


def _eval(model, dataloader, loss, cuda, verbose=False):
    model.eval()
    loss_total = MetricDict()
    outputs = []
    attns = []
    for data in dataloader:
        if cuda:
            data = (d.cuda() for d in data)
        data, time, weekday, target = data
        output = model(data, time, weekday)
        if isinstance(output, tuple):
            output, attn = output
            if verbose:
                outputs.append(output.detach().cpu().numpy())
                attns.append(attn.detach().cpu().numpy())
            else:
                del attn
        loss_total += loss(output, target)
    if verbose:
        output = np.concatenate(outputs, 0)
        attn = np.concatenate(attns, 0)
        return loss_total, output, attn
    return loss_total


if __name__ == '__main__':
    import fire
    fire.Fire()
