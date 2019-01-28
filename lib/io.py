import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.time_series import TimeSeries
from lib.loaders import get_loader


def load_data(config):
    df = get_loader(config.dataset).load_ts(config.freq)
    df = _filter_df(df, config.bday, config.start, config.end)
    df_train, df_val, df_test = _split_dataset(df, config.train_ratio, config.test_ratio)
    mean, std = df_train.mean().values, df_train.std().values

    dataset_train, dataset_valid, dataset_test = (
        TimeSeries(df, mean, std, config.history, config.horizon)
        for df in (df_train, df_val, df_test)
    )

    data_train = DataLoader(dataset_train, config.batch_size, True)
    data_validation = DataLoader(dataset_valid, config.batch_size)
    data_test = DataLoader(dataset_test, config.batch_size)
    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)

    return data_train, data_validation, data_test, mean, std


def _filter_df(df, bday, start, end):
    time_filter = (df.index.hour >= start) & (df.index.hour < end)
    if bday:
        bday_filter = df.index.weekday < 5
        return df[time_filter & bday_filter]
    else:
        return df[time_filter]


def _split_dataset(df, train_ratio=0.7, test_ratio=0.2):
    # calculate dates
    days = len(np.unique(df.index.date))
    days_train = pd.Timedelta(days=round(days * train_ratio))
    days_test = pd.Timedelta(days=round(days * test_ratio))
    date_train = df.index[0].date() + days_train
    date_test = df.index[-1].date() - days_test
    # select df
    dateindex = df.index.date
    df_train = df[dateindex < date_train]
    df_val = df[(dateindex >= date_train) & (dateindex < date_test)]
    df_test = df[dateindex >= date_test]
    return df_train, df_val, df_test


def load_adj(dataset):
    loader = get_loader(dataset)
    if dataset == 'LA':
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


def load_adj_long(dataset):
    loader = get_loader(dataset)
    dist = loader.load_dist().values
    dist = graph.digitize_dist(dist)
    if dataset.startswith('BJ'):
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


def load_distant_mask(dataset):
    adj = load_adj(dataset)
    mask = adj == 0
    return mask
