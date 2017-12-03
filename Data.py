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


def load_flow_data(affix='O'):
    filepath = DATA_PATH + 'O' + '.csv'
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
    return flow


def load_flow(gran=15, start_time=6):
    assert gran in [5, 10, 15, 20, 30, 60]
    steps = gran // 5
    origin = load_flow_data('O').astype(float).as_matrix()
    destination = load_flow_data('D').astype(float).as_matrix()
    flow = np.concatenate((origin, destination), -1)

    # sum up steps interval
    flow = flow.reshape((DAYS, -1, flow.shape[-1]))
    for i in range(flow.shape[1] - steps):
        flow[:, i, :] = flow[:, i:i + steps, :].sum(axis=1)
    flow = flow[:, :-steps]

    # trim the head
    head = start_time * 60 // 5
    flow = flow[:, head:]

    # trim the tail
    tail = flow.shape[1] % steps
    flow = flow[:, :-tail] if tail else flow

    # normalization
    flow = flow.reshape((-1, flow.shape[-1]))
    flow_mean = flow.mean(0)
    flow_std = flow.std(0)
    flow = (flow - flow_mean) / (flow_std + EPS)
    flow = flow.reshape((DAYS, -1, flow.shape[-1]))

    # slicing flow
    flow = np.concatenate([flow[:, i::steps, :] for i in range(steps)])

    # compute daytimes
    nday, ntime, _ = flow.shape
    daytimes = np.zeros((nday, ntime, 2)).astype(int)
    for iday in range(nday):
        daytimes[iday, :, 0] = (iday % DAYS - WEEKDAY) % 7
    for itime in range(ntime):
        daytimes[:, itime, 1] = itime

    return flow, daytimes, flow_mean, flow_std


def load_flow_seq(gran=15, start_time=6):
    flow, daytimes, flow_mean, flow_std = load_flow(gran)
    inputs = flow[:, :-1]
    targets = flow[:, 1:]
    daytimes = daytimes[:, :-1]
    return inputs, targets, daytimes, flow_mean, flow_std


def load_flow_img(gran=15, past=16, future=4, start_time=6):
    flow, daytimes, flow_mean, flow_std = load_flow(gran)

    nday, ntime, _ = flow.shape
    inputs, targets, daytimes_img = [], [], []
    for d in range(nday):
        for t in range(ntime - future):
            if t < past:
                # pad zeros as past
                padding = np.zeros((past - t, cols))
                feature = np.vstack((padding, flow[d, :t]))
            else:
                feature = flow[d, t - past:t]
            inputs.append(feature)
            targets.append(flow[d, t:t + future, :])
            daytimes_img.append(daytimes[d, t])
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    daytimes_img = np.asarray(daytimes_img)
    return inputs, targets, daytimes_img, flow_mean, flow_std
