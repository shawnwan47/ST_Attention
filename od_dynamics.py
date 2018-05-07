import argparse

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize

from lib import config
from lib import Loader


args = argparse.ArgumentParser()
config.add_data(args)
args = args.parse_args()
config.update_data(args)

assert args.dataset in ['BJ_highway', 'BJ_metro']


def load_od():
    if args.dataset == 'BJ_highway': loader = Loader.BJLoader('highway')
    else: loader = Loader.BJLoader('metro')
    return loader.load_ts_od()


def change_freq(OD):
    names = OD.index.names
    return OD.groupby([pd.Grouper(level=names[0], freq=freq),
                       names[1], names[2]]).sum()


def get_week(datetime):
    week = datetime.weekofyear == (datetime.weekofyear[len(datetime) // 2] + 1)
    hour = (datetime.hour >= args.start) & (datetime.hour < args.end)
    return datetime[week & hour]


def get_week_(datetime, period):
    assert period in ['last', 'day', 'week']
    if period == 'last':
        return datetime - (datetime[1] - datetime[0])
    elif period == 'day':
        return datetime - pd.Timedelta(days=1)
    elif period == 'week':
        return datetime - pd.Timedelta(days=7)


def compute_dynamics(ods, ods_, shape):
    return [od_distance(ods[idx], ods_[idx_], shape)
            for idx, idx_ in zip(
                ods.index.levels[0],
                ods_.index.levels[0])]


def od_distance(od, od_, shape):
    pk = od.to_coo()[0]
    qk = od_.to_coo()[0]
    pk = coo_matrix((pk.data, (pk.row, pk.col)), shape=shape)
    qk = coo_matrix((qk.data, (qk.row, qk.col)), shape=shape)
    p_sum = pk.sum(1)
    pk = normalize(pk, norm='l1', axis=1)
    qk = normalize(qk, norm='l1', axis=1)
    distances = hellinger_distance(pk, qk)
    return (np.multiply(distances, p_sum)).sum() / p_sum.sum()


def hellinger_distance(pk, qk):
    BC = pk.multiply(qk).sqrt().sum(1)
    return np.sqrt(1 - BC)


if __name__ == '__main__':
    OD = load_od()
    datetime = OD.index.levels[0]
    shape = (args.nodes // 2, args.nodes // 2)
    week = get_week(datetime)
    week_ = get_week_(week, 'last')
    idx = pd.IndexSlice
    ods, ods_ = OD.loc[idx[week, :, :]], OD.loc[idx[week_, :, :]]
    dynamics = compute_dynamics(ods, ods_, shape)
