import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from lib import Loader
from lib import IO
from lib.utils import aeq


class SparseDataset(Dataset):
    def __init__(self, coo_seqs, size):
        self.data = [self.parse_coo_seq(coo_seq, size) for coo_seq in coo_seqs]

    @staticmethod
    def parse_coo_seq(coo_seq, size):
        return [torch.sparse.FloatTensor(torch.LongTensor(i),
                                         torch.FloatTensor(v),
                                         torch.Size(size))
                for i, v in coo_seq]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class HybridDataset(Dataset):
    def __init__(self, tensor_dataset, list_dataset):
        assert len(tensor_dataset) == len(list_dataset)
        self.tensor_dataset = tensor_dataset
        self.list_dataset = list_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        return (*self.tensor_dataset[index], *self.list_dataset[index])


def _get_loader(dataset):
    if dataset == 'BJ_highway':
        loader = Loader.BJLoader('highway')
    elif dataset == 'BJ_metro':
        loader = Loader.BJLoader('metro')
    elif dataset == 'LA':
        loader = Loader.LALoader()
    return loader


def dataset_to_dataloader(data_train, data_valid, data_test, batch_size):
    return (DataLoader(dataset, batch_size, shuffle=i==0, pin_memory=True)
            for i, dataset in enumerate(data_train, data_valid, data_test))


def load_dataset(dataset, freq='5min', start=0, end=24, past=12, future=12):
    loader = _get_loader(dataset)
    ts = IO.TimeSeries(loader.load_ts(freq), start, end, past, future)
    mean = torch.FloatTensor(ts.mean)
    std = torch.FloatTensor(ts.std)

    data_train, data_valid, data_test = (
        TensorDataset(torch.FloatTensor(data[0]),
                      torch.LongTensor(data[1]),
                      torch.FloatTensor(data[2]))
        for data in (ts.data_train, ts.data_valid, ts.data_test)
    )

    return data_train, data_valid, data_test, mean, std


def load_dataset_od(
    dataset, node_count, freq='5min', start=0, end=24, past=12, future=12):
    data_train, data_valid, data_test, mean, std = load_dataset(
        dataset, freq, start, end, past, future
    )

    loader = _get_loader(dataset)
    od = loader.load_ts_od('DO', freq)
    ts = IO.SparseTimeSeries(od, start, end, past, future)
    od_train, od_valid, od_test = (
        SparseDataset(coo_seqs, (node_count, node_count))
        for coo_seqs in (ts.data_train, ts.data_valid, ts.data_test)
    )

    data_train, data_valid, data_test = (
        HybridDataset(data1, data2)
        for data1, data2 in zip([data_train, data_valid, data_test],
                                [od_train, od_valid, od_test])
    )

    return data_train, data_valid, data_test, mean, std


def load_adj(dataset, cuda=False):
    adj = _get_loader(dataset).load_adj()
    adj = torch.FloatTensor(adj)
    return adj.cuda() if cuda else adj


def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
