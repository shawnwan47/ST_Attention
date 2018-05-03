from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from geopy.distance import geodesic

from constants import DATA_PATH, BJ_HIGHWAY_PATH, BJ_METRO_PATH, LA_PATH
from lib.graph import floyd


class BJLoader:
    def __init__(self, dataset='highway'):
        self._path = Path(DATA_PATH)
        self._path /= BJ_HIGHWAY_PATH if dataset == 'highway' else BJ_METRO_PATH
        self._node = self._path / 'STATION.txt'
        self._link = self._path / 'LINK.txt'
        self._link_raw = self._path / 'LINK_RAW.txt'
        self._dist = self._path / 'dist.csv'
        self._hop = self._path / 'hop.csv'
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

    def load_ts(self, freq='5min'):
        o = self._load_ts('O').loc[:, self._idx]
        d = self._load_ts('D').loc[:, self._idx]
        return pd.concat((o, d), axis=1).resample(freq).sum()

    def load_hop(self):
        if self._hop.exists():
            hop = pd.read_csv(self._hop, index_col=0)
            hop.columns = [int(col) for col in hop.columns]
        else:
            link = self.load_link()
            idx = np.unique(link)
            hop = pd.DataFrame(2**31, index=idx, columns=idx)
            for i in range(link.shape[0]):
                i0, i1 = link[i, 0], link[i, 1]
                hop.loc[i0, i1] = link.loc[i1, i0] = 1
            for i in idx:
                hop.loc[i, i] = 0
            hop = floyd(hop)
            hop.to_csv(self._hop, index=True)
        return hop.loc[self._idx, self._idx].as_matrix()

    def load_dist(self):
        if self._dist.exists():
            dist = pd.read_csv(self._dist, index_col=0)
            dist.columns = [int(col) for col in dist.columns]
        else:
            node = self.load_node()
            link = self.load_link()
            idx = np.unique(link)
            dist = pd.DataFrame(float('inf'), index=idx, columns=idx)
            for i in range(link.shape[0]):
                i0, i1 = link[i, 0], link[i, 1]
                try:
                    loc0 = (node.LAT[i0], node.LON[i0])
                    loc1 = (node.LAT[i1], node.LON[i1])
                    dist.loc[i0, i1] = geodesic(loc0, loc1).kilometers
                except KeyError:
                    continue
            for i in idx:
                dist.loc[i, i] = 0
            dist = floyd(dist)
            dist.to_csv(self._dist, index=True)
        return dist.loc[self._idx, self._idx].as_matrix()

    def load_ts_od(self, od='OD', freq='15min'):
        assert od in ['OD', 'DO']
        filepath = self._ts_od if od == 'OD' else self._ts_do
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


class LALoader:
    def __init__(self):
        self._path = Path(DATA_PATH) / LA_PATH
        self._ids, self.id_to_idx, self.adj = pickle.load(
            open(self._path / 'adj_mx.pkl', 'rb'))
        self._node = self._path / 'graph_sensor_locations.csv'
        self._ts = self._path / 'df_highway_2012_4mon_sample.h5'
        self._link = self._path / 'distances_la_2012.csv'

    def load_ts(self, freq='5min'):
        ts = pd.read_hdf(self._ts).loc[:, self._ids]
        ts[ts==0.] = np.nan
        return ts.resample(freq).mean()

    def load_adj(self):
        return self.adj

    def load_node(self):
        return pd.read_csv(self._node, index_col=0)

    def load_link(self):
        return pd.read_csv(self._link)
