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


def load_dist(recalc=False):
    filepath = DATA_PATH + 'DIST.csv'
    ret_idx = load_idx()
    if os.path.exists(filepath) and recalc == False:
        dist = pd.read_csv(filepath, index_col=0)
        dist.columns = list(map(int, dist.columns))
    else:
        link = load_link()
        idx = np.unique(link)
        dist = pd.DataFrame(100, index=idx, columns=idx)
        for i in range(link.shape[0]):
            d = 1
            if link[i, 0] not in ret_idx or link[i, 1] not in ret_idx:
                d = 0.5
            dist.loc[link[i, 0], link[i, 1]] = d
            dist.loc[link[i, 1], link[i, 0]] = d
        for i in idx:
            dist.loc[i, i] = 0
        for k in idx:
            for i in idx:
                for j in idx:
                    tmp = dist.loc[i, k] + dist.loc[k, j]
                    if dist.loc[i, j] > tmp:
                        dist.loc[i, j] = tmp
        dist.to_csv(filepath, index=True)
    return dist.loc[ret_idx, ret_idx].as_matrix()


def load_od():
    idx = load_idx()
    od = pd.read_csv(DATA_PATH + 'OD.csv', index_col=0)
    od.index.name = ''
    od.columns = list(map(int, od.columns))
    od = od.loc[idx, idx].as_matrix()
    od += od.transpose()
    od /= (od.sum(0) + EPS)
    od = (od + od.transpose()) / 2
    return od


def load_adj(jump=5, contrib=0.01):
    dist = load_dist() <= jump
    od = load_od() >= contrib
    adj = dist + od
    adj = np.vstack([adj, adj])
    adj = np.hstack([adj, adj])
    return adj.astype(int)


def load_daytime(resolution=15):
    flow = load_flow()
    day = flow.index.map(lambda x: x.weekday())
    hour = flow.index.map(lambda x: x.hour)
    minute = flow.index.map(lambda x: x.minute)
    time = hour * 4 + minute // resolution
    day = np.array(day).reshape(-1, 96, 1)
    time = np.array(day).reshape(-1, 96, 1)
    return day, time


def load_loc():
    loc, _, _ = load_flow_pixel()
    for i in range(loc.shape[-1]):
        loc[:, :, i] = i
    return loc


def load_flow_pixel(bits=64):
    idx = load_idx()
    origin = load_flow('O').loc[:, idx].astype(float).as_matrix()
    destination = load_flow('D').loc[:, idx].astype(float).as_matrix()

    flow = np.concatenate((origin, destination), -1)
    flow_min = np.min(flow, 0)
    flow -= flow_min
    flow_scale = (np.max(flow, 0) + 1e-3) / bits
    flow /= flow_scale
    flow = flow.reshape(-1, 96, flow.shape[-1]).astype(int)
    return flow, flow_min, flow_scale


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
