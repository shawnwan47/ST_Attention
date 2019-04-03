from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import geodesic

from constants import DATASET_PATH, METR_LA, PEMS_BAY, BJ_HIGHWAY, BJ_SUBWAY


def get_loader(dataset):
    if dataset in [METR_LA, PEMS_BAY]:
        return RoadTraffic(dataset)
    elif dataset in [BJ_HIGHWAY, BJ_SUBWAY]:
        return StationTraffic(dataset)
    else:
        raise(KeyError('no such dataset!'))



class RoadTraffic:
    def __init__(self, dataset):
        self._path = Path(DATASET_PATH) / dataset
        self._adj = self._path / 'adj_mx.pkl'
        self.ids, self.id_to_idx, self.adj = pickle.load(
            open(self._path / 'adj_mx.pkl', 'rb'))
        self.adj[self.adj < 0.1] = 0
        self._prep_ids()
        self._node = self._path / 'sensors.csv'
        self._ts = self._path / 'ts.h5'
        self._link = self._path / 'distances.csv'
        self._dist = self._path / 'dist.csv'

    def _prep_ids(self):
        self.ids = sorted(self.ids, key=lambda x: self.id_to_idx[x])
        self.ids = [int(x) for x in self.ids]

    def load_node(self):
        return pd.read_csv(self._node, index_col=0)

    def load_link(self):
        return pd.read_csv(self._link)

    def load_ts(self, freq='5min'):
        ts = pd.read_hdf(self._ts)
        ts.columns = [int(col) for col in ts.columns]
        ts = ts.loc[:, self.ids]
        ts[ts==0.] = np.nan
        return ts.resample(freq).mean()

    def load_adj(self):
        return self.adj


class StationTraffic:
    def __init__(self, dataset):
        self._path = Path(DATASET_PATH) / dataset
        self._node = self._path / 'node.csv'
        self._link_raw = self._path / 'link_raw.csv'
        self._link = self._path / 'link.csv'
        if not self._link.exists():
            self._calc_link()
        self._hop = self._path / 'hop.csv'
        self._dist = self._path / 'dist.csv'
        self._ts_o = self._path / 'O.csv'
        self._ts_d = self._path / 'D.csv'
        self._ts_od = self._path / 'OD.csv'
        self._ts_do = self._path / 'DO.csv'
        self._od = self._path / 'ODSUM.csv'
        self.ids = self._load_ids()
        self.id_to_idx = {id: i for i, id in enumerate(self.ids)}

    def _load_ids(self):
        node = self._load_node()
        node_ids = set(node.index)
        link_ids = list(np.unique(self._load_link_raw()))
        ts_ids = [*self._load_ts('O').columns, *self._load_ts('D').columns]
        node_ids = node_ids.intersection(*[link_ids, ts_ids])
        ids = node.loc[node_ids].sort_values(['route', 'station']).index
        return ids

    def _load_node(self):
        return pd.read_csv(self._node, index_col=0)

    def _load_ts(self, od='D'):
        filepath = self._ts_o if od is 'O' else self._ts_d
        ts = pd.read_csv(filepath, index_col=0, parse_dates=True, dtype=float)
        ts.columns = [int(col) for col in ts.columns]
        return ts

    def _load_link_raw(self):
        return pd.read_csv(self._link_raw, dtype=int)

    def _calc_link(self):
        link = self._load_link_raw()
        node = self._load_node()
        pos = node.apply(lambda x: (x['latitude'], x['longitude']), axis=1)
        def dist(i, j):
            if i in node.index and j in node.index:
                return geodesic(pos[i], pos[j]).km
            return 0.
        link['cost'] = link.apply(lambda x: dist(x['from'], x['to']), axis=1)
        link.to_csv(self._link, index=False)

    def load_node(self):
        node = self._load_node().loc[self.ids]
        node.index = [self.id_to_idx[i] for i in node.index]
        return node

    def load_link(self):
        link = pd.read_csv(self._link)

    def _load_ts_o(self, freq):
        return self._load_ts('O').loc[:, self.ids].resample(freq).sum()

    def _load_ts_d(self, freq):
        return self._load_ts('D').loc[:, self.ids].resample(freq).sum()

    def load_ts(self, freq='15min'):
        o = self._load_ts_o(freq)
        d = self._load_ts_d(freq)
        ts = pd.concat((o, d), axis=1).fillna(0)
        ts = ts.resample(freq).sum()
        return ts

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
        _, entry, exit = ret.index.levels
        entry = [self.id_to_idx[ent] for ent in entry]
        exit = [self.id_to_idx[exi] for exi in exit]
        ret.index.set_levels(entry, level=1, inplace=True)
        ret.index.set_levels(exit, level=2, inplace=True)
        return ret

    def load_od(self):
        if self._od_sum.exists():
            ret = pd.read_csv(self._od_sum, index_col=0)
            ret.columns = [int(col) for col in ret.columns]
        else:
            od = self.load_ts_od(freq='1d').groupby(['Entry', 'Exit']).sum()
            od = od.unstack().fillna(0)
            ret = pd.DataFrame(0, index=self.ids, columns=self.ids)
            ret.loc[od.index, od.columns] = od
            ret.to_csv(self._od_sum, index=True)
        return ret
