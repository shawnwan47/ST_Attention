import os

import pandas as pd
import numpy as np

from Consts import *


def load_station():
    ret = pd.read_csv(DATA_PATH + 'STATION.csv', index_col=0)
    flow = load_flow_data()
    return ret.loc[flow.columns, :]


def load_adj_link():
    link = np.genfromtxt(DATA_PATH + 'LINK.txt', dtype=int)
    station = load_station()
    idx = station.index
    ret = pd.DataFrame(0, index=idx, columns=idx)
    for i in range(link.shape[0]):
        if link[i, 0] in idx and link[i, 1] in idx:
            ret.loc[link[i, 0], link[i, 1]] = 1
    return ret


def load_adj_od(contrib=0.01):
    od_name = DATA_PATH + 'OD.csv'
    od = pd.read_csv(od_name, index_col=0)
    od.index.name = ''
    od.columns = list(map(int, od.columns))
    station = load_station()
    od = od.loc[station.index, station.index]
    od = od + od.transpose()
    return od / od.sum(0) >= contrib


def load_adj(contrib=0.01):
    link = load_adj_link()
    od = load_adj_od()
    adj = (od + link + np.eye(len(od))) > 0
    adj = np.vstack([adj, adj])
    adj = np.hstack([adj, adj])
    return adj.astype(int)


def load_flow_data(affix='O'):
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
    flow.columns = list(map(int, flow.columns))
    return flow


def load_flow_highway():
    o = load_flow_data('O')
    d = load_flow_data('D')
    day = o.index.map(lambda x: x.weekday())
    hour = o.index.map(lambda x: x.hour)
    minute = o.index.map(lambda x: x.minute)
    time = hour * 4 + minute // 15

    o = o.astype(float).as_matrix()
    d = d.astype(float).as_matrix()
    flow = np.concatenate((o, d), -1)
    daytime = np.vstack((np.array(day), np.array(time))).T

    # normalization
    flow_mean = flow.mean(0)
    flow_std = flow.std(0) + EPS
    flow = (flow - flow_mean) / flow_std

    # reshape
    flow = flow.reshape(-1, 96, flow.shape[-1])
    daytime = daytime.reshape(-1, 96, 2)

    return flow, daytime, flow_mean, flow_std


def load_flow_metro(resolution=15, start_time=5, end_time=23):
    assert resolution in [5, 10, 15, 20, 30, 60]
    steps = resolution // 5
    o = load_flow_data('O').astype(float).as_matrix()
    d = load_flow_data('D').astype(float).as_matrix()
    flow = np.concatenate((o, d), -1)

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
