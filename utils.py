import os
import time
import math

import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable

from Constants import *


def load_adj_spatial():
    return np.genfromtxt(DATA_PATH + 'link.txt')


def load_adj_od(od_name=DATA_PATH + 'od_al.csv'):
    ret = pd.read_csv(od_name, index_col=0)
    ret.columns = list(map(int, ret.columns))
    ret.index.name = ''
    return ret


def load_flow_data(affix='D', minutes=15):
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

    steps = minutes // 5
    # sum up steps interval
    flow = flow.reshape((DAYS, -1, flow.shape[-1]))
    for day in range(DAYS):
        for i in range(flow.shape[0] - steps):
            flow[day, i, :] = flow[day, i:i + steps, :].sum(axis=0)
    flow = flow.reshape((-1, flow.shape[-1]))

    # trim the tail
    tail = flow.shape[0] % steps
    return flow if tail == 0 else flow[:-tail]


# data for model
def load_flow(minutes=15, history=24, future=8):
    o, d = load_flow_data('O', minutes), load_flow_data('D', minutes)
    flow = np.hstack((o, d))

    cols = flow.shape[-1]
    slices = minutes // 5

    # normalization
    flow_mean, flow_std = flow.mean(axis=0), flow.std(axis=0)
    flow = (flow - flow_mean) / (flow_std + EPS)

    # slice flow
    flow = flow.reshape((DAYS, -1, cols))
    flow = [flow[:, i::slices, :] for i in range(slices)]

    features, labels, days, times = [], [], [], []
    for day in range(DAYS):
        weekday = (day - WEEKDAY) % 7
        for f in flow:
            for t in range(history, f.shape[1] - future):
                features.append(f[day, t - history:t, :])
                labels.append(f[day, t:t + future, :])
                days.append(weekday)
                times.append(t - history)
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


def get_batch(features, labels, bsz):
    idx = np.random.randint(0, features.shape[0], bsz)
    var_features = np2torch(features[idx])
    var_labels = np2torch(labels[idx])
    return var_features, var_labels


def np2torch(flow):
    var_flow = Variable(torch.FloatTensor(flow))
    if USE_CUDA:
        var_flow = var_flow.cuda()
    return var_flow.transpose(0, 1)


if __name__ == '__main__':
    features, labels, days, times, flow_mean, flow_std = load_flow()
    pass
