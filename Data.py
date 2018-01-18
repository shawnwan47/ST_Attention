import os
import pickle as pk

import pandas as pd
import numpy as np

from Consts import *


def load_idx():
    filepath = DATA_PATH + 'idx.pk'
    if os.path.exists(filepath):
        idx = pk.load(open(filepath, 'rb'))
    else:
        station = load_station(False)
        station_idx = set(station.index)
        flow_idx = set(load_flow(clean=False).columns)
        link_idx = set(np.unique(load_link()))
        od_idx = set(load_od(False).index)
        for idx in [flow_idx, link_idx, od_idx]:
            station_idx.intersection_update(idx)
        idx = station.loc[station_idx].sort_values(['ROUTE', 'STATION']).index
        idx = list(idx)
        pk.dump(idx, open(filepath, 'wb'))
    return idx


def load_station(clean=True):
    if not clean:
        return pd.read_csv(DATA_PATH + 'STATION.txt', index_col=0)
    else:
        return load_station(clean=False).loc[load_idx()]


def load_link(raw=False):
    filename = 'LINK.txt' if not raw else 'LINK_RAW.txt'
    return np.genfromtxt(DATA_PATH + filename, dtype=int)


def load_od(clean=True):
    if clean==False:
        od = pd.read_csv(DATA_PATH + 'OD.csv', index_col=0)
    else:
        idx = load_idx()
        od = pd.read_csv(DATA_PATH + 'OD.csv', index_col=0)
        od.index.name = ''
        od.columns = list(map(int, od.columns))
        od = od.loc[idx, idx].as_matrix()
    return od

def load_flow(affix='D', clean=True):
    filepath = DATA_PATH + affix + '.csv'
    flow = pd.read_csv(filepath, index_col=0, parse_dates=True)
    flow.columns = list(map(int, flow.columns))
    if not clean:
        return flow
    else:
        return flow.loc[:, load_idx()]


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
    orig, _, _, dest, _, _ = load_flow_pixel()
    for i in range(orig.shape[-1]):
        orig[:, :, i] = i
        dest[:, :, i] = i
    dest += dest.shape[-1]
    return orig, dest


def load_flow_pixel(bits=64):
    idx = load_idx()
    orig = load_flow('O').loc[:, idx].astype(float).as_matrix()
    dest = load_flow('D').loc[:, idx].astype(float).as_matrix()

    def scale_flow(flow):
        flow_min = np.min(flow, 0)
        flow -= flow_min
        flow_scale = (np.max(flow, 0) + 1e-3) / bits
        flow /= flow_scale
        flow = flow.reshape(-1, 96, flow.shape[-1]).astype(int)
        return flow, flow_min, flow_scale

    orig, orig_min, orig_scale = scale_flow(orig)
    dest, dest_min, dest_scale = scale_flow(dest)
    return orig, orig_min, orig_scale, dest, dest_min, dest_scale
