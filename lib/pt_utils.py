import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from lib import Loader
from lib.IO import TimeSeries
from lib.utils import aeq


class SparseDataset(Dataset):
    def __init__(self, coo, size):
        self.data = [torch.sparse.FloatTensor(torch.LongTensor(i),
                                              torch.FloatTensor(v),
                                              torch.Size(size)) for i, v in coo]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return data[index]


class HybridDataset(Dataset):
    def __init__(self, *datasets):
        aeq(*[len(dataset) for dataset in datasets])
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        return (*dataset[index] for dataset in self.datasets)


def get_dataset(dataset, freq, start, end, past, future, bsz, cuda):
    # dataset
    freq = str(freq) + 'min'
    if dataset == 'BJ_highway':
        loader = Loader.BJLoader('highway')
    elif dataset == 'BJ_metro':
        loader = Loader.BJLoader('metro')
    elif dataset == 'LA':
        loader = Loader.LALoader()
    ts = loader.load_ts(freq)

    # time series
    io = TimeSeries(ts, start, end, past, future)

    dataset_tvt = [TensorDataset(torch.FloatTensor(data[0]),
                                 torch.LongTensor(data[1]),
                                 torch.FloatTensor(data[2]))
                   for data in (ts.data_train, ts.data_valid, ts.data_test)]

    data_train, data_valid, data_test = (
        DataLoader(dataset, batch_size=bsz, pin_memory=cuda, shuffle=i==0)
        for i, dataset in enumerate(dataset_tvt)
    )

    # adj
    adj_sp = sp.sparse.coo_matrix(adj)
    i = torch.LongTensor([adj_sp.row, adj_sp.col])
    v = torch.FloatTensor(adj_sp.data)
    adj = torch.sparse.FloatTensor(i, v)

    # mean std adj cuda
    mean = torch.FloatTensor(io.mean)
    std = torch.FloatTensor(io.std)
    if cuda:
        mean, std, adj = mean.cuda(), std.cuda(), adj.cuda()
    return data_train, data_valid, data_test, mean, std, adj


def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
