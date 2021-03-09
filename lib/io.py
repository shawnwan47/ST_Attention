import numpy as np
import pandas as pd
import torch

from lib.time_series import TimeSeries
from lib.loaders import get_loader
from lib import graph

from constants import PEMS_BAY, METR_LA, BJ_SUBWAY, BJ_HIGHWAY


def load_dataset(config):
    loader = get_loader(config.dataset)
    ts = loader.load_ts()
    node = loader.load_node()

    filter = _datetime_filter(ts.index, config.bday, config.start, config.end)
    ts = ts[filter]
    idx_train, idx_valid, idx_test = _split_dataset(ts.index, config.train_ratio, config.test_ratio)
    ts_train, ts_valid, ts_test = ts[idx_train], ts[idx_valid], ts[idx_test]
    mean, std = ts_train.mean().values, ts_train.std().values

    dataset_train, dataset_valid, dataset_test = (
        TimeSeries(ts, mean, std, config.history, config.horizon)
        for ts in (ts_train, ts_valid, ts_test)
    )

    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
    coordinates = torch.Tensor(node[['latitude', 'longitude']].values)
    if config.dataset in [BJ_SUBWAY, BJ_HIGHWAY]:
        coordinates = torch.cat([coordinates, coordinates], dim=0)

    return dataset_train, dataset_valid, dataset_test, mean, std, coordinates


def _datetime_filter(idx, bday, start, end):
    filter = (idx.hour >= start) & (idx.hour < end)
    if bday:
        bday_filter = idx.weekday < 5
        filter &= bday_filter
    return filter


def _split_dataset(idx, train_ratio=0.7, test_ratio=0.2):
    # calculate dates
    days = len(np.unique(idx.date))
    days_train = pd.Timedelta(days=round(days * train_ratio))
    days_test = pd.Timedelta(days=round(days * test_ratio))
    date_train = idx[0].date() + days_train
    date_test = idx[-1].date() - days_test
    # select ts
    idx_train = idx.date < date_train
    idx_valid = (idx.date >= date_train) & (idx.date < date_test)
    idx_test = idx.date >= date_test
    return idx_train, idx_valid, idx_test


def load_adj(dataset):
    loader = get_loader(dataset)
    if dataset in [METR_LA, PEMS_BAY]:
        adj = loader.load_adj()
    else:
        dist = loader.load_dist().values
        od = loader.load_od().values
        dist = graph.calculate_dist_adj(dist)
        od, do = graph.calculate_od_adj(od)
        adj0 = np.hstack((dist, od))
        adj1 = np.hstack((do, dist))
        adj = np.vstack((adj0, adj1))
    return torch.FloatTensor(adj)


def load_adj_mask(dataset):
    adj = load_adj(dataset)
    return torch.ByteTensor(adj <= 0.01)


def load_adj_long(dataset):
    loader = get_loader(dataset)
    dist = loader.load_dist().values
    dist = graph.digitize_dist(dist)
    if dataset in [BJ_SUBWAY, BJ_HIGHWAY]:
        od = loader.load_od().values
        od, do = graph.digitize_od(od)
        od += dist.max() + 1
        do += od.max() + 1
        adj0 = np.hstack((dist, od))
        adj1 = np.hstack((do, dist))
        adj = np.vstack((adj0, adj1))
        mask = (adj == dist.max()) | (adj == od.min()) | (adj == do.min())
    else:
        adj = dist
        mask = dist == dist.max()
    adj = torch.LongTensor(adj)
    mask = torch.ByteTensor(mask.astype(int))
    return adj, mask


def gen_subsequent_time(time, length):
    return torch.stack([time + i + 1 for i in range(length)], 1)


def gen_subsequent_mask(length=24):
    mask = np.triu(np.ones((length, length)), k=1).astype('uint8')
    return torch.from_numpy(mask)
