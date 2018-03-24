from pathlib import Path
import pickle as pk

import pandas as pd
import numpy as np


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
            dist = pd.read_csv(filepath.name, index_col=0)
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
            dist.to_csv(filepath.name, index=True)
        return dist.as_matrix()

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


class TrafficFlow:
    def __init__(self, dataset='highway',
                 freq=15, start=360, past=120, future=60,
                 inp='OD', out='OD'):
        assert freq in [5, 10, 15, 20, 30, 60]
        assert past <= start
        self.start = start // freq
        self.past = past // freq
        self.future = future // freq
        self.freq = str(freq) + 'T'
        # load data
        flow = self.loadFlow(dataset)
        self.start_day = flow.index[0].weekday()
        # normalize
        self.flow, self.mean, self.std = self.normalize(flow)

    def loadFlow(self, dataset):
        loader = Loader(dataset)
        flow_in = loader.load_flow('O', self.freq)
        flow_out = loader.load_flow('D', self.freq)
        return pd.concat((flow_in, flow_out), axis=1)

    def normalize(self, flow):
        days = (flow.index[-1].date() - flow.index[0].date()).days + 1
        mean, std = flow.mean(), flow.std() + 1e-8
        flow = (flow - mean) / std
        flow = flow.as_matrix().reshape((days, -1, flow.shape[1]))
        return flow, mean.as_matrix(), std.as_matrix()


class SpatialData(TrafficFlow):
    def __init__(self, dataset='highway',
                 freq=15, start=360, past=120, future=60,
                 inp='OD', out='OD'):
        super().__init__(dataset=dataset,
                         freq=freq, start=start, past=past, future=future,
                         inp=inp, out=out)
        # flow: num_day x num_time x num_loc x window
        self.data_num, self.targets = self.getFlowIO()
        # data_categorical: num_day x num_time x num_loc x 3
        self.data_cat = self.getCategorical()

    def getFlowIO(self):
        # ret: num_day x num_time x num_loc x window
        num_slots = self.flow.shape[1]
        num_time = num_slots - self.future - self.start
        # [num_day x num_loc x window]
<<<<<<< HEAD
        flow = self.flow.transpose(0, 2, 1)
        flow_i = [flow[:, :, self.start + i - self.past:self.start + i]
                  for i in range(num_time)]
        flow_o = [flow[:, :, self.start + i:self.start + i + self.future]
                  for i in range(num_time)]
        flow_i = np.stack(flow_i, axis=1)
        flow_o = np.stack(flow_o, axis=1)
=======
        self.flow = self.flow.transpose(0, 2, 1)
        flow_i = np.stack(
            [self.flow[:, :, self.start + i - self.past:self.start + i]
             for i in range(num_time)], axis=1)
        flow_o = np.stack(
            [self.flow[:, :, self.start + i:self.start + i + self.future]
             for i in range(num_time)], axis=1)
>>>>>>> 356a299cdfbf52d52135b8c1d8b408efaa058e11
        return flow_i, flow_o

    def getCategorical(self):
        num_day, num_time, num_loc, _ = self.data_num.shape
        day = np.arange(num_day).reshape(num_day, 1, 1)
        day = (day + self.start_day) % 7
        time = np.arange(num_time).reshape(1, num_time, 1)
        loc = np.arange(num_loc).reshape(1, 1, num_loc)
        ret = np.broadcast_arrays(day, time, loc)
        return np.stack(ret, -1)


class TemporalData(TrafficFlow):
<<<<<<< HEAD
    def __init__(self, **args):
        super().__init__(**args)
=======
    def __init__(self, dataset='highway',
                 freq=15, start=360, past=120, future=60,
                 inp='OD', out='OD'):
        super().__init__(dataset=dataset,
                         freq=freq, start=start, past=past, future=future,
                         inp=inp, out=out)
>>>>>>> 356a299cdfbf52d52135b8c1d8b408efaa058e11
        self.data, self.targets = self.getFlowIO()

    def getFlowIO(self):
        '''
        Temporal traffic flow within time span of a week
        '''

        pass
