from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import geodesic

from constants import DATA_PATH, BJ_HIGHWAY_PATH, BJ_METRO_PATH, LA_PATH
from lib import graph


class LoaderBase:
    def __init__(self):
        raise NotImplementedError

    def load_ts(self, freq):
        raise NotImplementedError

    def load_adj(self):
        raise NotImplementedError

    def load_node(self):
        return pd.read_csv(self._node, index_col=0)

    def load_link(self):
        return pd.read_csv(self._link)

    def load_dist(self):
        if self._dist.exists():
            dist = pd.read_csv(self._dist, index_col=0)
            dist.columns = [int(col) for col in dist.columns]
        else:
            G = graph.build_graph(self.load_link())
            dist = graph.graph_dist(G)
            dist = dist.loc[self.ids, self.ids]
            dist.to_csv(self._dist, index=True)
        return dist

    def load_hop(self):
        if self._hop.exists():
            hop = pd.read_csv(self._hop, index_col=0, dtype=int)
            hop.columns = [int(col) for col in hop.columns]
        else:
            G = graph.build_graph(self.load_link())
            hop = graph.graph_hop(G)
            hop = hop.loc[self.ids, self.ids]
            hop.to_csv(self._hop, index=True)
        return hop


class BJLoader(LoaderBase):
    def __init__(self, dataset='highway'):
        self._path = Path(DATA_PATH)
        self._path /= BJ_HIGHWAY_PATH if dataset == 'highway' else BJ_METRO_PATH
        self._node = self._path / 'STATION.txt'
        self._link_raw = self._path / 'link_raw.csv'
        self._link = self._path / 'link.csv'
        if not self._link.exists():
            self.calc_link()
        self._hop = self._path / 'hop.csv'
        self._dist = self._path / 'dist.csv'
        self._ts_o = self._path / 'O.csv'
        self._ts_d = self._path / 'D.csv'
        self._ts_od = self._path / 'OD.csv'
        self._ts_do = self._path / 'DO.csv'
        self._od_sum = self._path / 'ODSUM.csv'
        self.ids = self._load_ids()
        self.id_to_idx = {id: i for i, id in enumerate(self.ids)}

    def _load_ids(self):
        node = self.load_node()
        node_ids = set(node.index)
        link_ids = list(np.unique(self.load_link_raw()))
        ts_ids = [*self._load_ts('O').columns, *self._load_ts('D').columns]
        node_ids = node_ids.intersection(*[link_ids, ts_ids])
        ids = node.loc[node_ids].sort_values(['ROUTE', 'STATION']).index
        return ids

    def _load_ts(self, od='D'):
        filepath = self._ts_o if od is 'O' else self._ts_d
        ts = pd.read_csv(filepath, index_col=0, parse_dates=True, dtype=float)
        ts.columns = [int(col) for col in ts.columns]
        return ts

    def load_link_raw(self):
        return pd.read_csv(self._link_raw, dtype=int)

    def calc_link(self):
        link = self.load_link_raw()
        node = self.load_node()
        pos = node.apply(lambda x: (x['latitude'], x['longitude']), axis=1)
        def dist(i, j):
            if i in node.index and j in node.index:
                return geodesic(pos[i], pos[j]).km
            return 0.
        link['cost'] = link.apply(lambda x: dist(x['from'], x['to']), axis=1)
        link.to_csv(self._link, index=False)

    def load_ts(self, freq='5min'):
        o = self._load_ts('O').reindex(columns=self.ids)
        d = self._load_ts('D').reindex(columns=self.ids)
        ts = pd.concat((o, d), axis=1).fillna(0)
        ts = ts.resample(freq).sum()
        return ts

    def load_ts_od(self, od='OD', freq='5min'):
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
        return ret.as_matrix()

    def load_adj(self):
        dist = graph.calculate_dist_adj(self.load_dist())
        od = self.load_od()
        od = od / od.sum() * dist.sum()
        adj0, adj1 = np.hstack((dist, od)), np.hstack((od, dist))
        adj = np.vstack((adj0, adj1))
        return adj


class LALoader(LoaderBase):
    def __init__(self):
        self._path = Path(DATA_PATH) / LA_PATH
        self.ids, self.id_to_idx, self.adj = pickle.load(
            open(self._path / 'adj_mx.pkl', 'rb'))
        self.adj[self.adj < 0.1] = 0
        self._prep_ids()
        self._node = self._path / 'graph_sensor_locations.csv'
        self._ts = self._path / 'df_highway_2012_4mon_sample.h5'
        self._link = self._path / 'distances_la_2012.csv'
        self._dist = self._path / 'dist.csv'
        self._hop = self._path / 'hop.csv'

    def _prep_ids(self):
        self.ids = sorted(self.ids, key=lambda x: self.id_to_idx[x])
        self.ids = [int(x) for x in self.ids]

    def load_ts(self, freq='5min'):
        ts = pd.read_hdf(self._ts)
        ts.columns = [int(col) for col in ts.columns]
        ts = ts.loc[:, self.ids]
        ts[ts==0.] = np.nan
        return ts.resample(freq).mean()

    def load_adj(self):
        return self.adj
