import os
import pickle as pk

import pandas as pd
import numpy as np


class Loader(object):
    def __init__(self, dataset='highway'):
        if dataset == 'highway':
            self.DATA_PATH = 'data/highway/'
        else:
            self.DATA_PATH = 'data/metro/'
        self.idx = self._load_idx()

    def _load_idx(self):
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


class Dataset(object):
    def __init__(self, dataset='highway', freq=15,
                 start=360, history=120, future=60,
                 inp='OD', out='OD'):
        assert freq in [5, 10, 15, 20, 30, 60]
        start //= freq
        history //= freq
        future //= freq
        if dataset == 'highway':
            self.DAYS_TRAIN = 120
            self.DAYS_TEST = 30
        else:
            self.DAYS_TRAIN = 14
            self.DAYS_TEST = 4
        # load data
        flow = self.loadFlow(dataset, freq)
        start_day = flow.index[0].weekday()
        # normalize
        flow, self.mean, self.std = self.normalize(flow)
        # flow: num_day x num_time x num_loc x window
        flow_i, flow_o = self.getFlowIO(flow, start, history, future)
        # daytimeloc: num_day x num_time x num_loc x 3
        daytimeloc = self.getCategorical(flow_i, start_day)
        # query, context, flow_o
        self.context_numerical = self.getIO(flow_i, inp)
        self.context_categorical = self.getIO(daytimeloc, inp)
        self.query_numerical = self.getIO(flow_i, out)
        self.query_categorical = self.getIO(daytimeloc, out)
        self.targets = self.getIO(flow_o, out)

    def loadFlow(self, dataset, freq):
        loader = Loader(dataset)
        flow_in, flow_out = loader.load_flow_in(), loader.load_flow_out()
        flow = pd.concat((flow_in, flow_out), axis=1)
        return flow.resample(str(freq) + 'T').sum()

    def normalize(self, flow):
        days = (flow.index[-1].date() - flow.index[0].date()).days + 1
        mean, std = flow.mean(), flow.std() + 1e-8
        flow = (flow - mean) / std
        flow = flow.as_matrix().reshape((days, -1, flow.shape[1]))
        return flow, mean, std

    def getFlowIO(self, flow, start, history, future):
        # ret: num_day x num_time x num_loc x window
        num_slots = flow.shape[1]
        num_time = num_slots - future - start
        # [num_day x num_loc x window]
        flow = flow.transpose(0, 2, 1)
        flow_i = [flow[:, :, start + i - history:start + i]
                  for i in range(num_time)]
        flow_o = [flow[:, :, start + i:start + i + future]
                  for i in range(num_time)]
        flow_i = np.stack(flow_i, axis=1)
        flow_o = np.stack(flow_o, axis=1)
        return flow_i, flow_o

    def getCategorical(self, flow, start_day):
        num_day, num_time, num_loc, _ = flow.shape
        day = np.arange(num_day).reshape(num_day, 1, 1)
        day = (day + start_day) % 7
        time = np.arange(num_time).reshape(1, num_time, 1)
        loc = np.arange(num_loc).reshape(1, 1, num_loc)
        ret = np.broadcast_arrays(day, time, loc)
        return np.stack(ret, -1)

    def getIO(self, data, od):
        # data: num_day, num_time, num_loc, num_features
        assert len(data.shape) == 4
        stations = data.shape[2] / 2
        assert stations == int(stations)
        if od is 'O':
            return data[:, :, :stations]
        if od is 'D':
            return data[:, :, -stations:]
        else:
            return data

    def getTrainValidTest(self, data):
        assert len(data.shape) == 4
        _, _, num_loc, num_features = data.shape
        shape = (-1, num_loc, num_features)
        data_train = data[:self.DAYS_TRAIN].reshape(shape)
        data_valid = data[self.DAYS_TRAIN:-self.DAYS_TEST].reshape(shape)
        data_test = data[-self.DAYS_TEST:].reshape(shape)
        return data_train, data_valid, data_test
