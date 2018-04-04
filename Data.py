from pathlib import Path
import pickle as pk

import pandas as pd
import numpy as np
from sklearn import preprocessing


class Loader:
    def __init__(self, dataset='highway'):
        self.DATA_PATH = Path('data') / dataset
        self.idx = self._load_idx()

    def _load_idx(self):
        station = self.load_station_raw()
        station_idx = set(station.index)
        link_idx = list(np.unique(self.load_link()))
        flow_idx = [*self._load_flow('O').columns, *self._load_flow('D').columns]
        station_idx = station_idx.intersection(*[link_idx, flow_idx])
        idx = station.loc[station_idx].sort_values(['ROUTE']).index
        return idx

    def _load_flow(self, od='D'):
        flow = pd.read_csv(self.DATA_PATH / (od + '.csv'),
                           index_col=0, parse_dates=True)
        flow.columns = list(map(int, flow.columns))
        return flow

    def load_station_raw(self):
        return pd.read_csv(self.DATA_PATH / 'STATION.txt', index_col=0)

    def load_station(self):
        station = self.load_station_raw()
        return station.loc[self.idx]

    def load_link(self):
        return np.genfromtxt(self.DATA_PATH / 'LINK.txt', dtype=int)

    def load_link_raw(self):
        return np.genfromtxt(self.DATA_PATH / 'LINK_RAW.txt', dtype=int)

    def load_flow(self, od='O', freq='5T'):
        flow = self._load_flow(od)
        for col in self.idx.drop(flow.columns):
            flow[col] = 0
        flow = flow.asfreq(freq)
        return flow[self.idx]

    def load_dist(self):
        filepath = self.DATA_PATH / 'DIST.csv'
        if filepath.exists():
            dist = pd.read_csv(filepath, index_col=0)
            dist.columns = [int(col) for col in dist.columns]
        else:
            link = self.load_link()
            idx = np.unique(link)
            dist = pd.DataFrame(2**31, index=idx, columns=idx)
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
        return dist.loc[self.idx, self.idx].as_matrix()

    def load_od(self, od='OD', freq='5T'):
        assert od in ['OD', 'DO']
        ret = pd.read_csv(self.DATA_PATH / (od + '.csv'),
                          index_col=[0, 1, 2],
                          parse_dates=[0],
                          squeeze=True)
        names = ret.index.names
        ret = ret.groupby([pd.Grouper(level=names[0], freq=freq),
                           names[1], names[2]]).sum()
        return ret

    def load_od_sum(self):
        filepath = self.DATA_PATH / 'ODSUM.csv'
        if filepath.exists():
            ret = pd.read_csv(filepath, index_col=0)
            ret.columns = [int(col) for col in ret.columns]
        else:
            od = self.load_od(freq='1d').groupby(['Entry', 'Exit']).sum()
            od = od.unstack().fillna(0)
            ret = pd.DataFrame(0, index=self.idx, columns=self.idx)
            ret.loc[od.index, od.columns] = od
            ret.to_csv(filepath, index=True)
        return ret.as_matrix()


class TrafficFlow:
    def __init__(self, dataset='highway',
                 freq=15, start=360, past=120, future=60):
        assert freq in [5, 10, 15, 20, 30, 60]
        assert past <= start
        self.start = start // freq
        self.past = past // freq
        self.future = future // freq
        self.freq = str(freq) + 'T'
        # load data
        flow = self.loadFlow(dataset)
        self.start_day = flow.index[0].weekday()
        days = (flow.index[-1].date() - flow.index[0].date()).days + 1
        # scale data
        scaler = preprocessing.StandardScaler().fit(flow)
        self.mean, self.scale = scaler.mean_, scaler.scale_
        self.flow = scaler.transform(flow).reshape((days, -1, flow.shape[1]))

    def loadFlow(self, dataset):
        loader = Loader(dataset)
        flow_in = loader.load_flow('O', self.freq)
        flow_out = loader.load_flow('D', self.freq)
        return pd.concat((flow_in, flow_out), axis=1)


class SpatialTraffic(TrafficFlow):
    def __init__(self, dataset='highway',
                 freq=15, start=360, past=120, future=60):
        super().__init__(dataset=dataset,
                         freq=freq, start=start, past=past, future=future)
        # flow: num_day x num_time x num_loc x window
        self.data_num, self.targets = self.get_data_num()
        # data_categorical: num_day x num_time x num_loc x 3
        self.data_cat = self.get_data_cat()
        del self.flow

    def get_data_num(self):
        # ret: num_day x num_time x num_loc x window
        num_slots = self.flow.shape[1]
        num_time = num_slots - self.future - self.start
        # [num_day x num_loc x window]
        self.flow = self.flow.transpose(0, 2, 1)
        flow_i = np.stack(
            [self.flow[:, :, self.start + i - self.past:self.start + i]
             for i in range(num_time)], axis=1)
        flow_o = np.stack(
            [self.flow[:, :, self.start + i:self.start + i + self.future]
             for i in range(num_time)], axis=1)
        return flow_i, flow_o

    def get_data_cat(self):
        num_day, num_time, num_loc, _ = self.data_num.shape
        day = np.arange(num_day).reshape(num_day, 1, 1)
        day = (day + self.start_day) % 7
        time = np.arange(num_time).reshape(1, num_time, 1)
        loc = np.arange(num_loc).reshape(1, 1, num_loc)
        ret = np.broadcast_arrays(day, time, loc)
        return np.stack(ret, -1)
