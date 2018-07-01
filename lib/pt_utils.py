import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from lib import Loader
from lib import IO
from lib import graph
from lib.utils import aeq
from constants import EPS


def _get_loader(dataset):
    if dataset == 'BJ_highway':
        loader = Loader.BJLoader('highway')
    elif dataset == 'BJ_metro':
        loader = Loader.BJLoader('metro')
    elif dataset == 'LA':
        loader = Loader.LALoader()
    return loader


def load_dataset(dataset, freq, history, horizon, batch_size):
    df = _get_loader(dataset).load_ts(freq)
    io = IO.TimeSeries(df, history, horizon)
    mean = numpy_to_torch(io.mean)
    std = numpy_to_torch(io.std)

    datasets = (
        TensorDataset(*[numpy_to_torch(data) for data in data_tuple])
        for data_tuple in io.data
    )

    data_loaders = (
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=i==0)
        for i, dataset in enumerate(datasets)
    )

    return data_loaders, mean, std


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
    loader = _get_loader(dataset)
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
    loader = _get_loader(dataset)
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


def mask_target(output, target):
    mask = ~torch.isnan(target)
    return output.masked_select(mask), target.masked_select(mask)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def numpy_to_torch(arr):
    assert isinstance(arr, np.ndarray)
    if arr.dtype == np.dtype(int):
        return torch.LongTensor(arr)
    return torch.FloatTensor(arr)


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


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


class Rescaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input * (self.std + EPS)) + self.mean
