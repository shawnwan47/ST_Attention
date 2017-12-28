import os

import pandas as pd
import numpy as np

from Consts import *


def load_station():
    return pd.read_csv(DATA_PATH + 'STATION.csv', index_col=0)


def load_link():
    return np.genfromtxt(DATA_PATH + 'LINK.txt', dtype=int)


def load_flow(affix='O'):
    filepath = DATA_PATH + affix + '.csv'
    flow = pd.read_csv(filepath, index_col=0, parse_dates=True)
    flow.columns = list(map(int, flow.columns))
    return flow


def load_idx():
    station = set(load_station().index)
    link = set(np.unique(load_link()))
    flow = set(load_flow().columns)
    return station.intersection(link).intersection(flow)


def load_dist():
    filepath = DATA_PATH + 'DIST.csv'
    if os.path.exists(filepath):
        dist = pd.read_csv(filepath, index_col=0)
        dist.columns = list(map(int, dist.columns))
    else:
        link = load_link()
        idx = np.unique(link)
        dist = pd.DataFrame(100, index=idx, columns=idx)
        for i in range(link.shape[0]):
            dist.loc[link[i, 0], link[i, 1]] = 1
        for i in idx:
            dist.loc[i, i] = 0
        for k in idx:
            for i in idx:
                for j in idx:
                    tmp = dist.loc[i, k] + dist.loc[k, j]
                    if dist.loc[i, j] > tmp:
                        dist.loc[i, j] = tmp
        dist.to_csv(filepath, index=True)
    idx = load_idx()
    return dist.loc[idx, idx].as_matrix()


def load_od():
    idx = load_idx()
    od = pd.read_csv(DATA_PATH + 'OD.csv', index_col=0)
    od.index.name = ''
    od.columns = list(map(int, od.columns))
    od = od.loc[idx, idx].as_matrix()
    od = od + od.transpose()
    od = od / (od.sum(0) + EPS)
    od = (od + od.transpose()) / 2
    return od


def load_adj(jump=5, contrib=0.01):
    dist = load_dist() <= jump
    od = load_od() >= contrib
    adj = dist + od
    adj = np.vstack([adj, adj])
    adj = np.hstack([adj, adj])
    return adj.astype(int)


def load_flow_highway():
    idx = load_idx()
    origin = load_flow('O').loc[:, idx]
    destination = load_flow('D').loc[:, idx]
    day = origin.index.map(lambda x: x.weekday())
    hour = origin.index.map(lambda x: x.hour)
    minute = origin.index.map(lambda x: x.minute)
    time = hour * 4 + minute // 15

    origin = origin.astype(float).as_matrix()
    destination = destination.astype(float).as_matrix()
    flow = np.concatenate((origin, destination), -1)
    daytime = np.vstack((np.array(day), np.array(time))).T

    # normalization
    flow_mean = flow.mean(0)
    flow_std = flow.std(0) + EPS
    flow = (flow - flow_mean) / flow_std
    flow_diff = np.diff(flow, axis=0)
    flow_diff = np.append(flow_diff[[0]], flow_diff, 0)

    # reshape
    flow = flow.reshape(-1, 96, flow.shape[-1])
    flow_diff = flow_diff.reshape(-1, 96, flow_diff.shape[-1])
    daytime = daytime.reshape(-1, 96, 2)

    return flow, flow_diff, flow_mean, flow_std, daytime


def load_flow_metro(resolution=15, start_time=5, end_time=23):
    assert resolution in [5, 10, 15, 20, 30, 60]
    steps = resolution // 5
    origin = load_flow('O').astype(float).as_matrix()
    destination = load_flow('D').astype(float).as_matrix()
    flow = np.concatenate((origin, destination), -1)

    # sum up steps interval
    flow = flow.reshape((DAYS, -1, flow.shape[-1]))
    for i in range(flow.shape[1] - steps):
        flow[:, i] = flow[:, i:i + steps].sum(axis=1)
    flow = flow[:, :-steps]

    # trim the head and tail
    head = start_time * 60 // 5
    tail = end_time * 60 // 5
    flow = flow[:, head:tail]

    # trim the mod and slice flow
    res = flow.shape[1] % steps
    flow = flow[:, :-res] if res else flow
    flow = np.concatenate([flow[:, i::steps, :] for i in range(steps)], 0)

    # normalization
    flow = flow.reshape((-1, flow.shape[-1]))
    flow_mean = flow.mean(0)
    flow_std = flow.std(0)
    flow = (flow - flow_mean) / (flow_std + EPS)
    flow = flow.reshape((DAYS * steps, -1, flow.shape[-1]))

    # compute daytime
    nday, ntime, _ = flow.shape
    daytime = np.zeros((nday, ntime, 2)).astype(int)
    for iday in range(nday):
        daytime[iday, :, 0] = (iday % DAYS - WEEKDAY) % 7
    for itime in range(ntime):
        daytime[:, itime, 1] = itime

    return flow, daytime, flow_mean, flow_std
