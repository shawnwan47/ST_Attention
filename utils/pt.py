import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import loader
from utils.io import TimeSeries


class TrafficDataset(Dataset):
    def __init__(self, data_num, data_cat, targets):
        self.data_num = torch.FloatTensor(data_num)
        self.data_cat = torch.LongTensor(data_cat)
        self.targets = torch.FloatTensor(targets)

    def __getitem__(self, index):
        return self.data_num[index], self.data_cat[index], self.targets[index]

    def __len__(self):
        return self.targets.size(0)


def get_dataset(dataset, freq=15, start=6, end=23, past=12, future=12,
                batch_size=16, cuda=False):
    if dataset is 'BJ_highway':
        dataset = loader.BJLoader('highway')
    elif dataset is 'BJ_metro':
        dataset = loader.BJLoader('metro')
    else:
        dataset = loader.LALoader()
    ts, adj = dataset.load_ts(freq), dataset.load_adj()
    io = TimeSeries(ts)
    data_num, data_cat, targets = io.gen_daily_seq_io()
    data_num, data_cat, targets = (io.train_val_test_split(data)
                                   for data in (data_num, data_cat, targets)))

    data_train, data_valid, data_test = (
        TrafficDataset(data_num[i], data_cat[i], targets[i]) for i in range(3))

    data_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=cuda)
    data_valid = DataLoader(data_valid, batch_size=batch_size, pin_memory=cuda)
    data_test = DataLoader(data_test, batch_size=batch_size, pin_memory=cuda)

    mean = torch.FloatTensor(io.mean)
    scale = torch.FloatTensor(io.scale)
    adj = torch.FloatTensor(adj)
    if cuda:
        mean, scale, adj = mean.cuda(), scale.cuda(), adj.cuda()
    return data_train, data_valid, data_test, mean, scale, adj


def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
