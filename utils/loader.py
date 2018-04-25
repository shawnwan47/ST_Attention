from pathlib import Path
import pickle

import pandas as pd
import numpy as np


DATA_PATH = Path('data/dataset')


class BJLoader:
    def __init__(self, dataset='highway'):
        self._path = DATA_PATH / ('BJ_' + dataset)
        self._node = self._path / 'STATION.txt'
        self._link = self._path / 'LINK.txt'
        self._link_raw = self._path / 'LINK_RAW.txt'
        self._dist = self._path / 'DIST.csv'
        self._ts_o = self._path / 'O.csv'
        self._ts_d = self._path / 'D.csv'
        self._ts_od = self._path / 'OD.csv'
        self._ts_do = self._path / 'DO.csv'
        self._od_sum = self._path / 'ODSUM.csv'
        self._idx = self._load_idx()

    def _load_idx(self):
        node = self.load_node(raw=True)
        node_idx = set(node.index)
        link_idx = list(np.unique(self.load_link()))
        ts_idx = [*self._load_ts('O').columns, *self._load_ts('D').columns]
        node_idx = node_idx.intersection(*[link_idx, ts_idx])
        idx = node.loc[node_idx].sort_values(['ROUTE', 'STATION']).index
        return idx

    def _load_ts(self, od='D'):
        filepath = self._ts_o if od is 'O' else self._ts_d
        ts = pd.read_csv(filepath, index_col=0, parse_dates=True)
        ts.columns = [int(col) for col in ts.columns]
        return ts

    def load_node(self, raw=False):
        station = pd.read_csv(self._node, index_col=0)
        return station if raw else station.loc[self._idx]

    def load_link(self, raw=False):
        filepath = self._link_raw if raw else self._link
        return np.genfromtxt(filepath, dtype=int)

    def load_ts(self, freq='5T'):
        o = self._load_ts('O').loc[:, self._idx]
        d = self._load_ts('D').loc[:, self._idx]
        return pd.concat((o, d), axis=1).asfreq(freq)

    def load_dist(self):
        if self._dist.exists():
            dist = pd.read_csv(self._dist, index_col=0)
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
            dist.to_csv(self._dist, index=True)
        return dist.loc[self._idx, self._idx].as_matrix()

    def load_ts_od(self, od='OD', freq='5T'):
        assert od in ['OD', 'DO']
        filepath = self._ts_od is od is 'OD' else self._ts_do
        ret = pd.read_csv(filepath,
                          index_col=[0, 1, 2],
                          parse_dates=[0],
                          squeeze=True)
        names = ret.index.names
        ret = ret.groupby([pd.Grouper(level=names[0], freq=freq),
                           names[1], names[2]]).sum()
        return ret

    def load_adj_od(self):
        if self._od_sum.exists():
            ret = pd.read_csv(self._od_sum, index_col=0)
            ret.columns = [int(col) for col in ret.columns]
        else:
            od = self.load_od(freq='1d').groupby(['Entry', 'Exit']).sum()
            od = od.unstack().fillna(0)
            ret = pd.DataFrame(0, index=self._idx, columns=self._idx)
            ret.loc[od.index, od.columns] = od
            ret.to_csv(self._od_sum, index=True)
        return ret.as_matrix()

    def load_adj(self, od_ratio=0.01, hops=5):
        od = self.load_adj_od()
        dist = self.load_dist()
        adj_od = (od / od.sum()) < od_ratio
        adj_dist = dist > hops
        od[adj_od], od[~adj_od] = 1, 0
        dist[adj_dist], dist[~adj_dist] = 1, 0
        adj = np.vstack((np.hstack((dist, od)), np.hstack((od, dist))))
        np.fill_diagonal(adj, 1)
        return adj


class LALoader:
    def __init__(self):
        self._path = DATA_PATH / 'LA_highway'
        self._node = self._path / 'graph_sensor_locations.csv'
        self._ts = self._path / 'df.h5'
        self._link = self._path / 'distances_la_2012.csv'
        self._ids, self.id_to_idx, self.adj = pickle.load(
            open(self._path / 'adj_mx.pkl', 'rb'))

    def load_ts(self, freq='5T'):
        return pd.read_hdf(self._ts).loc[:, self._ids].asfreq(freq)

    def load_adj(self):
        return self.adj

    def load_node(self):
        return pd.read_csv(self._node, index_col=0)

    def load_link(self):
        return pd.read_csv(self._link)
