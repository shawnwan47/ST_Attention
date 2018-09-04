import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize

from lib import config
from lib import Loader
from constants import RESULT_PATH, FIG_PATH
from lib.utils import cat_strs


args = argparse.ArgumentParser()
args.add_argument('-dataset', choices=['BJ_highway', 'BJ_metro'])
args = args.parse_args()

DATASET = args.dataset
NUM_NODES = 536 if DATASET is 'BJ_metro' else 264

FREQS = ['5min', '10min', '15min', '30min', '1h', '3h', '6h', '12h', '1d']
PERIODS = ['last', 'day', 'week']
PRE_OD = 'OD'
PRE_DOD = 'DOD'


################################################################################
# IO
################################################################################
def csvpath(*strs):
    name = cat_strs(*strs) + '.csv'
    return Path(RESULT_PATH) / DATASET / name


def pngpath(*strs):
    name = cat_strs(*strs) + '.png'
    return Path(FIG_PATH) / DATASET / name


def read_csv_series(path):
    return pd.read_csv(path,
                       header=None,
                       squeeze=True,
                       index_col=0,
                       parse_dates=True)


def load_od():
    if DATASET == 'BJ_highway': loader = Loader.BJLoader('highway')
    else: loader = Loader.BJLoader('metro')
    return loader.load_ts_od(od='OD'), loader.load_ts_od('DO')


def od_asfreq(od, freq):
    names = od.index.names
    datetime_group = pd.Grouper(level=names[0], freq=freq)
    od_freq = od.groupby([datetime_group, names[1], names[2]]).sum()
    return od_freq


def get_week_index(od):
    datetime = od.index.levels[0]
    return datetime[datetime.weekofyear == (datetime.weekofyear[0] + 2)]


def get_period_index(datetime, period):
    assert period in PERIODS
    if period == 'last':
        return datetime - (datetime[1] - datetime[0])
    elif period == 'day':
        return datetime - pd.Timedelta(days=1)
    elif period == 'week':
        return datetime - pd.Timedelta(days=7)


################################################################################
# temporal sparsity
################################################################################




################################################################################
# od distance
################################################################################
def od_distance(od, od_):
    if od.empty or od_.empty:
        return np.nan
    pdata = (od.values / od.values.sum()).tolist()
    prow = od.index.get_level_values(0).tolist()
    pcol = od.index.get_level_values(1).tolist()
    qdata = (od_.values / od_.values.sum()).tolist()
    qrow = od_.index.get_level_values(0).tolist()
    qcol = od_.index.get_level_values(1).tolist()
    shape = (NUM_NODES // 2, NUM_NODES // 2)
    pk = coo_matrix((pdata, (prow, pcol)), shape=shape)
    qk = coo_matrix((qdata, (qrow, qcol)), shape=shape)
    return hellinger_distance(pk, qk)


def hellinger_distance(pk, qk):
    BC = pk.multiply(qk).sqrt().sum()
    return np.sqrt(1 - BC)


################################################################################
# dump results
################################################################################

def calc_dump_results():
    od, do = load_od()
    for freq in FREQS:
        od_freq, do_freq = asfreq(od, freq), asfreq(do, freq)
        print(f'Transformed OD to {freq}...')
        symmetric = calc_od_symmetric(od_freq, do_freq)
        symmetric.to_csv(csvpath(PRE_DOD, freq))
        for period in PERIODS:
            dynamic = calc_od_dynamic(od_freq, period)
            dynamic.to_csv(csvpath(PRE_OD, freq, period))


def calc_od_dynamic(od, period='last'):
    index = get_week_index(od)
    index_ = get_period_index(index, period)
    distances = [od_distance(od[idx], od[idx_])
                 for idx, idx_ in zip(index, index_)]
    return pd.Series(distances, index=index)


def calc_od_symmetric(od, do):
    index = get_week_index(od)
    distances = [od_distance(od[idx], do[idx]) for idx in index]
    return pd.Series(distances, index=index)


################################################################################
# PLOTTING
################################################################################

def plot_series(prefix, on=None):
    figpath = pngpath(prefix, on)
    fig = plt.figure()
    if on in FREQS:
        for period in PERIODS:
            read_csv_series(csvpath(prefix, on, period)).plot(label=period)
    else:
        for freq in FREQS:
            read_csv_series(csvpath(prefix, freq, on)).plot(label=freq)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


def plot_results():
    plot_series(PRE_DOD)
    for freq in FREQS:
        plot_series(PRE_OD, freq)
    for period in PERIODS:
        plot_series(PRE_OD, period)


if __name__ == '__main__':
    calc_dump_results()
    # plot_results()

    pass
