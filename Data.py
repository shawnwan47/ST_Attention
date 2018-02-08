import os
import pickle as pk

import pandas as pd
import numpy as np

from Consts import *


class Loader(object):
    def __init__(self, dataset):
        if dataset == 'highway':
            self.DATA_PATH = 'data/highway/'
        else:
            self.DATA_PATH = 'data/metro/'
        self.idx = self.__load_idx()

    def __load_idx(self):
        filepath = self.DATA_PATH + 'idx.pk'
        if os.path.exists(filepath):
            idx = pk.load(open(filepath, 'rb'))
        else:
            station = self.load_station_raw()
            station_idx = set(station.index)
            flow_idx = set(self._load_flow().columns)
            link_idx = set(np.unique(self.load_link()))
            for idx in [flow_idx, link_idx]:
                station_idx.intersection_update(idx)
            idx = station.loc[station_idx].sort_values(['ROUTE', 'STATION']).index
            pk.dump(idx, open(filepath, 'wb'))
        return idx

    def _load_flow(self, od='D'):
        flow = pd.read_csv(self.DATA_PATH + od + '.csv',
                           index_col=0, parse_dates=True)
        flow.columns = list(map(int, flow.columns))
        return flow

    def load_station(self):
        station = self.load_station_raw()
        return station.loc[self.idx]

    def load_station_raw(self):
        return pd.read_csv(self.DATA_PATH + 'STATION.txt', index_col=0)

    def load_link(self):
        return np.genfromtxt(self.DATA_PATH + 'LINK.txt', dtype=int)

    def load_link_raw(self):
        return np.genfromtxt(self.DATA_PATH + 'LINK_RAW.txt', dtype=int)

    def load_flow_in(self):
        flow = self._load_flow('O')
        return flow[self.idx]

    def load_flow_out(self):
        flow = self._load_flow('D')
        return flow[self.idx]

    def load_dist(self):
        filepath = DATA_PATH + 'DIST.csv'
        if os.path.exists(filepath):
            dist = pd.read_csv(filepath, index_col=0)
            dist.columns = list(map(int, dist.columns))
        else:
            link = self.load_link()
            idx = np.unique(link)
            dist = pd.DataFrame(100, index=idx, columns=idx)
            for i in range(link.shape[0]):
                dist.loc[link[i, 0], link[i, 1]] = 1
                dist.loc[link[i, 1], link[i, 0]] = 1
            for i in idx:
                dist.loc[i, i] = 0
            for k in idx:
                for i in idx:
                    for j in idx:
                        tmp = dist.loc[i, k] + dist.loc[k, j]
                        if dist.loc[i, j] > tmp:
                            dist.loc[i, j] = tmp
            dist.to_csv(filepath, index=True)
        return dist.as_matrix()




class Preprocesser(object):
    def __init__(self, dataset, freq):
        assert freq in [5, 10, 15, 20, 30, 60]
        io = ['O', 'D', 'OD']
        assert flow_in in io and flow_out in io
        if dataset == 'highway':
            self.DAYS_TRAIN = 120
            self.DAYS_TEST = 30
        else:
            self.DAYS_TRAIN = 14
            self.DAYS_TEST = 4
        loader = Loader(dataset)
        flow_in, flow_out = loader.load_flow_in(), loader.load_flow_out()
        flow = pd.concat((flow_in, flow_out), axis=1)
        self.mean, self.std = flow.mean(), flow.std() + 1e-8
        weekday = flow.index.map(lambda x: x.weekday())
        flow = (flow - self.mean) / self.std
        self.flow = flow.as_matrix().reshape((-1, 1440 // freq, flow.shape[1]))
        # IO
        data = self.getFlow(flow, flow_in)
        targets = self.getFlow(flow, flow_out)
        self.data_train, self.data_valid, self.data_test = self.split(data)
        self.targets_train, self.targets_valid, self.targets_test = self.split(targets)



if __name__ == '__main__':
    loader = Loader('highway')
