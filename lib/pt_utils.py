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


class MyDataset(Dataset):
    def __init__(self, data, targets):
        assert isinstance(data, dict)
        self.data = {name: numpy_to_torch(arr) for name, arr in data.items()}
        self.targets = numpy_to_torch(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        data = {name: tensor[index] for name, tensor in self.data.items()}
        target = self.targets[index]
        return data, targets


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


def dataset_to_dataloader(batch_size, *datasets):
    return (DataLoader(dataset, batch_size, shuffle=i==0)
            for i, dataset in enumerate(datasets))


def load_dataset(dataset, freq, history, horizon, data_source):
    df = _get_loader(dataset).load_ts(freq)
    io = IO.TimeSeries(df, history, horizon, data_source)
    mean = numpy_to_torch(io.mean)
    std = numpy_to_torch(io.std)

    data_train, data_valid, data_test = (
        TensorDataset(*[numpy_to_torch(data) for data in data_tuple])
        for data_tuple in (io.data_train, io.data_valid, io.data_test)
    )

    return data_train, data_valid, data_test, mean, std


def load_dataset_od(dataset, freq, history, horizon):
    data_train, data_valid, data_test, mean, std = load_dataset(args)

    loader = _get_loader(args.dataset)
    od = loader.load_ts_od('DO', args.freq)
    ts = IO.SparseTimeSeries(od, args.history, args.horizon)
    od_train, od_valid, od_test = (
        SparseDataset(coo_seqs, (args.node_count, args.node_count))
        for coo_seqs in (ts.data_train, ts.data_valid, ts.data_test)
    )

    data_train, data_valid, data_test = (
        HybridDataset(data1, data2)
        for data1, data2 in zip([data_train, data_valid, data_test],
                                [od_train, od_valid, od_test])
    )

    return data_train, data_valid, data_test, mean, std


def load_adj(dataset):
    adj = _get_loader(dataset).load_adj()
    adj = torch.FloatTensor(adj)
    return adj


def mask_target(output, target):
    mask = ~torch.isnan(target)
    return output.masked_select(mask), target.masked_select(mask)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def numpy_to_torch(arr):
    assert isinstance(arr, np.ndarray)
    if nparray.dtype == np.dtype(int):
        return torch.LongTensor(arr)
    return torch.FloatTensor(arr)
