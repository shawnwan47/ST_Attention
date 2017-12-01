import os

import pandas as pd
import numpy as np

from Consts import *


def load_adj_spatial():
    spatial = np.genfromtxt(DATA_PATH + 'link.txt', dtype=int)
    od = load_od()
    adj = od.copy().astype(int)
    adj[:] = 0
    for i in range(spatial.shape[0]):
        if spatial[i, 0] in od.columns and spatial[i, 1] in od.columns:
            adj.loc[spatial[i, 0], spatial[i, 1]] = 1
    return adj


def load_od(od_name=DATA_PATH + 'od_al.csv'):
    ret = pd.read_csv(od_name, index_col=0)
    ret.columns = list(map(int, ret.columns))
    ret.index.name = ''
    return ret


def od2graph(od, contrib=0.01):
    return od / od.sum(0) >= contrib


def load_adj(contrib=0.01):
    adj = load_adj_spatial().as_matrix()
    od = load_od().as_matrix()
    do = od2graph(od, contrib)
    od = od2graph(od.transpose(), contrib).transpose()
    oood = np.hstack((adj, od))
    dodd = np.hstack((do, adj))
    return np.vstack((oood, dodd))


def load_flow_data(affix='D', gran=15):
    assert gran in [5, 10, 15, 20, 30, 60]
    steps = gran // 5
    filepath = DATA_PATH + affix + '.csv'
    if os.path.exists(filepath):
        flow = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        data_files = sorted(os.listdir(DATA_PATH))
        data_files = list(filter(
            lambda x: x.startswith(affix + '_'), data_files))
        flow = pd.concat([
            pd.read_csv(DATA_PATH + x, index_col=0, parse_dates=True)
            for x in data_files])
        flow.to_csv(filepath, index=True)

    flow = flow.astype(float).as_matrix()

    # sum up steps interval
    flow = flow.reshape((DAYS, -1, flow.shape[-1]))
    for i in range(flow.shape[1] - steps):
        flow[:, i, :] = flow[:, i:i + steps, :].sum(axis=1)
    flow = flow[:, :-steps]

    # trim the head, start at 6
    head = 360 // gran
    flow = flow[:, head:, :]
    # trim the tail
    tail = flow.shape[1] % steps
    flow = flow[:, :-tail] if tail else flow

    # normalization
    flow = flow.reshape((-1, flow.shape[-1]))
    flow_mean = flow.mean(-1)
    flow_std = flow.std(-1)
    flow = (flow - flow_mean) / (flow_std + EPS)

    # slicing flow
    flow = flow.reshape((DAYS, -1, flow.shape[-1]))
    flow = [flow[:, i::steps, :] for i in range(steps)]

    # compute days and times
    nday, ntime, _ = flow.shape
    days, times = np.zeros((nday, ntime, 1)), np.zeros((nday, ntime))
    for iday in range(nday):
        day = (iday % DAYS - WEEKDAY) % 7
        for itime in range(ntime):
            pass

    return np.concatenate(flow), flow_mean, flow_std, days, times


# data for model
def load_flow(gran=15, past=24, future=8):
    o, o_mean, o_std = load_flow_data('O', gran)
    d, d_mean, d_std = load_flow_data('D', gran)
    flow = np.concatenate((o, d), -1)
    flow_mean = np.concatenate((o_mean, d_mean))
    flow_std = np.concatenate((o_std, d_std))

    days, length, stations = flow.shape

    features, labels, days, times = [], [], [], []
    start6 = 360 // gran  # start at 6
    for day in range(DAYS):
        weekday = (day - WEEKDAY) % 7
        for f in flow:
            for t in range(start6, f.shape[1] - future):
                if t < past:
                    # pad zeros as past
                    padding = np.zeros((past - t, cols))
                    feature = np.vstack((padding, f[day, :t]))
                else:
                    feature = f[day, t - past:t]
                features.append(feature)
                labels.append(f[day, t:t + future, :])
                days.append(weekday)
                times.append(t - past)
    features, labels = np.asarray(features), np.asarray(labels)
    days, times = np.asarray(days), np.asarray(times)
    return features, labels, days, times, flow_mean, flow_std


def split_dataset(data, unit):
    data_train = data[:unit * DAYS_TRAIN]
    data_valid = data[unit * DAYS_TRAIN:unit * -DAYS_TEST]
    data_test = data[unit * -DAYS_TEST:]
    return data_train, data_valid, data_test
