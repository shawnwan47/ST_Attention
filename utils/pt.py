import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import loader, io


class TrafficDataset(Dataset):
    def __init__(self, data_num, data_cat, targets):
        self.data_num = torch.FloatTensor(data_num)
        self.data_cat = torch.LongTensor(data_cat)
        self.targets = torch.FloatTensor(targets)

    def __getitem__(self, index):
        return self.data_num[index], self.data_cat[index], self.targets[index]

    def __len__(self):
        return self.targets.size(0)


def get_dataset(dataset, freq=15, start=6, end=23, batch_size=16, cuda=False):
    if dataset is 'BJ_highway':
        dataset = loader.BJLoader('highway')
    elif dataset is 'BJ_metro':
        dataset = loader.BJLoader('metro')
    else:
        dataset = loader.LALoader()
    ts, adj = dataset.load_ts(freq), dataset.load_adj()
    ts_io = io.TimeSeries(ts)
    data_num, data_cat, targets = ts_io.gen_daily_seq_io()
    data_num, data_cat, targets = (ts_io.train_val_test_split(data)
                                   for data in (data_num, data_cat, targets)))

    data_train, data_valid, data_test = (
        TrafficDataset(data_num[i], data_cat[i], targets[i]) for i in range(3))

    data_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_valid = DataLoader(data_valid, batch_size=batch_size)
    data_test = DataLoader(data_test, batch_size=batch_size)

    mean = torch.FloatTensor(ts_io.mean)
    scale = torch.FloatTensor(ts_io.scale)
    adj = torch.FloatTensor(adj)
    if cuda:
        mean, scale, adj = mean.cuda(), scale.cuda(), adj.cuda()
    return data_train, data_valid, data_test, mean, scale, adj


def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def get_mask_od(num_loc, od):
    assert od in ['OD', 'O', 'D']
    assert not num_loc % 2
    mask = torch.zeros(num_loc, num_loc)
    if od is 'O':
        mask[:, num_loc/2:] = 1
    if od is 'D':
        mask[:, :num_loc/2] = 1
    return mask.byte()


def get_mask_adj(dataset, od_ratio=0.01, hops=5):
    loader = Data.Loader(dataset)
    od = loader.load_od_sum()
    dist = loader.load_dist()
    mask_od = (od / od.sum()) < od_ratio
    mask_dist = dist > hops
    od[mask_od], od[~mask_od] = 1, 0
    dist[mask_dist], dist[~mask_dist] = 1, 0
    mask = np.vstack((np.hstack((dist, od)), np.hstack((od, dist))))
    np.fill_diagonal(mask, 1)
    return torch.ByteTensor(mask)


def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
