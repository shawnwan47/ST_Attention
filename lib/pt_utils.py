import numpy as np
import scipy as sp
import torch
from torch.utils.data import DataLoader

from lib.Loader import get_loader
from lib import IO
from lib import graph
from lib import Dataset
from lib.utils import aeq
from constants import EPS


def load_data(args):
    df = get_loader(args.dataset).load_ts(args.freq)
    df_train, df_valid, df_test, mean, std = IO.prepare_dataset(
        df=df,
        bday=args.bday,
        start=args.start,
        end=args.end
    )

    dataset_train, dataset_valid, dataset_test = (
        Dataset(df, mean, std, args.history, args.horizon)
        for df in (df_train, df_valid, df_test)
    )

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size)

    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)

    return loader_train, loader_valid, loader_test, mean, std


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


def mask_target(output, target):
    mask = ~torch.isnan(target)
    return output.masked_select(mask), target.masked_select(mask)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()
