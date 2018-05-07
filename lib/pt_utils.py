import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset, DataLoader

from lib import Loader
from lib.IO import TimeSeries


class MyDataset(Dataset):
    def __init__(self, data_num, data_cat, target):
        self.data_num = torch.FloatTensor(data_num)
        self.data_cat = torch.LongTensor(data_cat)
        self.target = torch.FloatTensor(target)

    def __getitem__(self, index):
        return (self.data_num[index], self.data_cat[index], self.target[index])

    def __len__(self):
        return self.target.size(0)


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

    dataset_tvt = [MyDataset(data[0], data[1], data[2]) for data in data_tvt]

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
