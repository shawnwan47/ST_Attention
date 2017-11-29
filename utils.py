import os
import time
import math

import pandas as pd
import numpy as np


from Constants import *


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


def od2graph(od, contrib=0.05):
    return od / od.sum(0) >= contrib


def load_adj():
    adj = load_adj_spatial().as_matrix()
    od = load_od().as_matrix()
    od = od2graph(od)
    oood = np.hstack((adj, od))
    dodd = np.hstack((od.transpose(), adj))
    return np.vstack((oood, dodd))


def load_flow_data(affix='D', granularity=15):
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

    flow = flow.as_matrix()

    steps = granularity // 5
    # sum up steps interval
    flow = flow.reshape((DAYS, -1, flow.shape[-1]))
    for day in range(DAYS):
        for i in range(flow.shape[0] - steps):
            flow[day, i, :] = flow[day, i:i + steps, :].sum(axis=0)
    flow = flow[:, :-steps]

    # trim the tail
    tail = flow.shape[1] % steps
    if tail:
        flow = flow[:, :-tail]
    return flow.reshape((-1, flow.shape[-1]))


# data for model
def load_flow(granularity=15, past=24, future=8):
    o, d = load_flow_data('O', granularity), load_flow_data('D', granularity)
    flow = np.hstack((o, d))

    cols = flow.shape[-1]
    slices = granularity // 5

    # normalization
    flow_mean, flow_std = flow.mean(axis=0), flow.std(axis=0)
    flow = (flow - flow_mean) / (flow_std + EPS)

    # slice flow
    flow = flow.reshape((DAYS, -1, cols))
    flow = [flow[:, i::slices, :] for i in range(slices)]

    features, labels, days, times = [], [], [], []
    start6 = 360 // granularity  # start at 6
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


def split_dataset(data):
    unit = data.shape[0] // DAYS
    data_train = data[:unit * DAYS_TRAIN]
    data_valid = data[unit * DAYS_TRAIN:unit * -DAYS_TEST]
    data_test = data[unit * -DAYS_TEST:]
    return data_train, data_valid, data_test


def timeSince(since, percent):
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

