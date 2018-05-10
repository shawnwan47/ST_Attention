import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from lib import Loader
from lib.IO import TimeSeries


class SparseDataset(Dataset):
    def __init__(self, coo, size):
        self.data = [torch.sparse.FloatTensor(torch.LongTensor(i),
                                              torch.FloatTensor(v),
                                              torch.Size(size)) for i, v in coo]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return data[index]


class HybridDataset():
    pass


def get_dataset(dataset, freq, start, end, past, future, bsz, cuda):
    # dataset
    freq = str(freq) + 'min'
    if dataset == 'BJ_highway':
        loader = Loader.BJLoader('highway')
    elif dataset == 'BJ_metro':
        loader = Loader.BJLoader('metro')
    elif dataset == 'LA':
        loader = Loader.LALoader()
    ts, adj = loader.load_ts(freq), loader.load_adj()

    # time series
    io = TimeSeries(ts)
    data_tvt = [io.gen_seq2seq_io(data, past, future, i==0)
                for i, data in enumerate(
                    [io.data_train, io.data_valid, io.data_test])]

    dataset_tvt = [TensorDataset(data[0], data[1], data[2]) for data in data_tvt]

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
