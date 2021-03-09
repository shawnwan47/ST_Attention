import numpy as np
import torch
from torch import tensor
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from lib import TimeSeriesLoss, MetricDict, Metric, mask_target
from lib.io import load_dataset
from builder import build_model


def _set_cuda(config):
    cuda = torch.cuda.is_available() & config.cuda
    if cuda:
        torch.cuda.set_device(config.gpuid)
        torch.cuda.manual_seed(config.seed)
        print(f'Using GPU: {config.gpuid}')
    else:
        torch.manual_seed(config.seed)
        print('Using CPU')
    return cuda


def _eval(model, dataloader, tsloss, cuda, verbose=False):
    model.eval()
    loss_eval = MetricDict()
    outputs = []
    for data in dataloader:
        if cuda:
            data = (d.cuda() for d in data)
        data, time, weekday, target = data
        output = model(data, time, weekday)
        outputs.append(output.detach().cpu().numpy())
        loss_eval += tsloss(output, target)
    return loss_eval, outputs if verbose else loss_eval


def _train(model, dataloader, criterion, optimizer, tsloss, cuda):
    model.train()
    loss_train = MetricDict()
    for data in dataloader:
        if cuda:
            data = (d.cuda() for d in data)
        data, time, weekday, target = data
        output = model(data, time, weekday)
        loss_train += tsloss(output, target)
        output, target = mask_target(output, target)
        score = criterion(output, target)
        optimizer.zero_grad()
        score.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return loss_train


def train(**kwargs):
    config = Config(**kwargs)
    cuda = _set_cuda(config)
    dataset_train, dataset_valid, dataset_test, mean, std, coordinates = load_dataset(config)
    loader_train = DataLoader(dataset_train, config.batch_size, True)
    loader_valid = DataLoader(dataset_valid, config.batch_size)
    loader_test = DataLoader(dataset_test, config.batch_size)
    model = build_model(config, mean, std, coordinates)
    criterion = getattr(nn, config.criterion)()
    optimizer = Adam(model.parameters(), weight_decay=config.weight_decay)
    tsloss = TimeSeriesLoss(metrics=config.metrics, horizons=config.horizons)
    tensorboard = SummaryWriter(config.path_tensorboard)
    data, time, weekday, _ = next(iter(loader_test))
    tensorboard.add_graph(model, [data, time, weekday])

    # early stopping initialization
    score_best = float('inf')
    trial = 0
    for epoch in range(config.epoches):
        loss_train = _train(model, loader_train, criterion, optimizer, tsloss, cuda)
        loss_valid = _eval(model, loader_valid, tsloss, cuda)
        score_train = tsloss.get_criterion(loss_train)
        score_valid = tsloss.get_criterion(loss_valid)
        tensorboard.add_scalars('loss', {'train': score_train, 'valid':score_valid}, epoch) 
        print(f'Epoch: {epoch}\nTrain: {loss_train}\nValid: {loss_valid}')
        # early stopping
        if score_valid < score_best:
            score_best = score_valid
            trial = 0
            torch.save(model.state_dict(), config.path_model)
        else:
            trial += 1
        if trial > config.patience:
            tensorboard.close()
            break
    model.load_state_dict(torch.load(config.path_model))
    loss_test = _eval(model, loader_test, tsloss, cuda, verbose=True)
    print(f'{config.path_model}:\n{loss_test}')


def test(**kwargs):
    config = Config(**kwargs)
    cuda = _set_cuda(config)
    _, _, dataset_test, mean, std, coordinates = load_dataset(config)
    dataloader = DataLoader(dataset_test, config.batch_size)
    model = build_model(config, mean, std, coordinates)
    model.load_state_dict(torch.load(config.path_model))
    tsloss = TimeSeriesLoss(metrics=config.metrics, horizons=config.horizons)
    loss_test, outputs = _eval(model, dataloader, tsloss, cuda, verbose=True)
    print(loss_test)
    output = outputs[:config.num_times]
    np.save(open(config.path_result.with_suffix('.output'), 'wb'), output)


if __name__ == '__main__':
    import fire
    fire.Fire()
