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
config.add_data(args)
args = args.parse_args()
config.update_data(args)

assert args.dataset in ['BJ_highway', 'BJ_metro']
FREQS = ['5min', '10min', '15min', '30min', '1h', '3h', '6h', '12h', '1d']
PERIODS = ['last', 'day', 'week']


def load_od(od='OD'):
    if args.dataset == 'BJ_highway': loader = Loader.BJLoader('highway')
    else: loader = Loader.BJLoader('metro')
    return loader.load_ts_od(od=od)


def asfreq(OD, freq):
    names = OD.index.names
    datetime_group = pd.Grouper(level=names[0], freq=freq)
    od_freq = OD.groupby([datetime_group, names[1], names[2]]).sum()
    print(f'Transformed OD to {freq}...')
    return od_freq


def get_week_index(datetime):
    week = datetime.weekofyear == (datetime.weekofyear[0] + 2)
    return datetime[week]


def get_period_index(datetime, period):
    assert period in PERIODS
    if period == 'last':
        return datetime - (datetime[1] - datetime[0])
    elif period == 'day':
        return datetime - pd.Timedelta(days=1)
    elif period == 'week':
        return datetime - pd.Timedelta(days=7)


################################################################################
# Computing
################################################################################
def compute_distances(od, index, od_=None, index_=None):
    if od_ is None:
        od_ = od
    if index_ is None:
        index_ = index
    return [od_distance(od[idx], od_[idx_])
            for idx, idx_ in zip(index, index_)]


def od_distance(od, od_):
    if od.empty or od_.empty:
        return np.nan
    pdata = (od.values / od.values.sum()).tolist()
    prow = od.index.get_level_values(0).tolist()
    pcol = od.index.get_level_values(1).tolist()
    qdata = (od_.values / od_.values.sum()).tolist()
    qrow = od_.index.get_level_values(0).tolist()
    qcol = od_.index.get_level_values(1).tolist()
    shape = (args.nodes // 2, args.nodes // 2)
    pk = coo_matrix((pdata, (prow, pcol)), shape=shape)
    qk = coo_matrix((qdata, (qrow, qcol)), shape=shape)
    distance = hellinger_distance(pk, qk)
    return distance


################################################################################
# dump results
################################################################################
def hellinger_distance(pk, qk):
    BC = np.multiply(pk, qk).sqrt().sum()
    return np.sqrt(1 - BC)


def get_csv_path(*features):
    filename = cat_strs(*features) + '.csv'
    path = Path(RESULT_PATH) / args.dataset / filename
    return path


def dump_od_dynamic(od, freq='1h', period='last'):
    path = get_csv_path(freq, period)
    index = get_week_index(od.index.levels[0])
    index_ = get_period_index(index, period)
    dynamic = compute_distances(od=od, index=index, index_=index_)
    sym_df = pd.Series(dynamic, index=index)
    sym_df.to_csv(path, index=True)
    return sym_df


def dump_od_symmetric(od, do, freq='1h'):
    path = get_csv_path('sym', freq)
    index = get_week_index(od.index.levels[0])
    symetric = compute_distances(od=od, od_=do, index=index)
    sym_df = pd.Series(symetric, index=index)
    sym_df.to_csv(path, index=True)
    return sym_df


def dump_results():
    od = load_od('OD')
    do = load_od('DO')
    for freq in FREQS:
        od_freq, do_freq = asfreq(od, freq), asfreq(do, freq)
        dump_od_symmetric(od_freq, do_freq, freq)
        for period in PERIODS:
            dump_od_dynamic(od_freq, freq, period)


################################################################################
# PLOTTING
################################################################################
def get_fig_path(*features):
    filename = cat_strs(*features) + '.png'
    return Path(FIG_PATH) / args.dataset / filename


def read_csv_series(path):
    return pd.read_csv(path,
                       header=None,
                       squeeze=True,
                       index_col=0,
                       parse_dates=True)


def plot_dynamic(od, freq, period=None, label=None):
    filepath = get_csv_path(od, freq, period)
    series = read_csv_series(filepath)
    series.plot(label=label)


def plot_freq(od, period=None):
    path = get_fig_path(od, period)
    # path = '/'.join(path.parts)
    plt.figure()
    plt.ylim(0, 1)
    for freq in FREQS:
        plot_dynamic(od, freq, period, label=freq)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_period(od, freq):
    path = get_fig_path(od, freq)
    # path = '/'.join(path.parts)
    fig = plt.figure()
    plt.ylim(0, 1)
    for i, period in enumerate(PERIODS):
        series = read_csv_series(get_csv_path(od, freq, period))
        series.plot(label=period)
        # axes[i].set_ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_results():
    plot_freq('sym')
    for freq in FREQS:
        plot_period('od', freq)
    for period in PERIODS:
        plot_freq('od', period)


if __name__ == '__main__':
    dump_results()
    plot_results()
