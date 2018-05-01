import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import Loader
from utils.IO import TimeSeries


class TrafficDataset(Dataset):
    def __init__(self, data_num, data_cat, target):
        self.data_num = torch.FloatTensor(data_num)
        self.data_cat = torch.LongTensor(data_cat)
        self.target = torch.FloatTensor(target)

    def __getitem__(self, index):
        return self.data_num[index], self.data_cat[index], self.target[index]

    def __len__(self):
        return self.target.size(0)


def get_dataset(dataset, freq=15, past=12, future=12, bsz=16, cuda=False):
    freq = str(freq) + 'min'
    if dataset == 'BJ_highway':
        dataset = Loader.BJLoader('highway')
    elif dataset == 'BJ_metro':
        dataset = Loader.BJLoader('metro')
    elif dataset == 'LA_highway':
        dataset = Loader.LALoader()
    ts, adj = dataset.load_ts(freq), dataset.load_adj()
    io = TimeSeries(ts)
    data_tvt = [io.gen_seq2seq_io(data, past, future, i==0)
                for i, data in enumerate(
                    [io.data_train, io.data_valid, io.data_test])]

    dataset_tvt = [TrafficDataset(data[0], data[1], data[2]) for data in data_tvt]

    dataloader_train, dataloader_valid, dataloader_test = (
        DataLoader(dataset, batch_size=bsz, pin_memory=cuda, shuffle=i==0)
        for i, dataset in enumerate(dataset_tvt)
    )

    mean = torch.FloatTensor(io.mean)
    std = torch.FloatTensor(io.std)
    adj = torch.FloatTensor(adj)
    if cuda:
        mean, std, adj = mean.cuda(), std.cuda(), adj.cuda()
    return dataloader_train, dataloader_valid, dataloader_test, mean, std, adj


def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
