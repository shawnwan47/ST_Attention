import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import Loader


def savefig(path):
    plt.tight_layout()
    plt.savefig(path + '.png')


class MapPainter:
    def __init__(self, dataset):
        loader = Loader(dataset)
        self.link = loader.load_link()
        self.station = loader.load_station()
        self.station_raw = loader.load_station_raw()

    def plot_road(self):
        plt.axis('off')
        station, link = self.station_raw, self.link
        for i in range(link.shape[0]):
            s, e = link[i, 0], link[i, 1]
            if s in station.index and e in station.index:
                plt.plot(station.loc[[s, e], 'LON'],
                         station.loc[[s, e], 'LAT'],
                         color='gray', linewidth=1)

    def plot_station(self, val=5, indices=None, scale=1):
        station = self.station.copy()
        if indices is not None:
            assert len(val) == len(indices)
            station = station.iloc[indices]
        plt.scatter(station['LON'], station['LAT'],
                    s=val*scale, alpha=0.5, edgecolors='none')


class RelationPainter(MapPainter):
    def __init__(self, dataset, relation, idx=None):
        '''
        relation: day x time x loc x loc
        '''
        assert len(relation.shape) == 4
        super().__init__(dataset)



class ODPlotter:
    def __init__(self, dataset):
        assert dataset in ['BJ_higway', 'BJ_metro']
        loader = Loader(dataset)
        self.station = loader.load_station()
        self.routes = self._get_routes(self.station)

    @staticmethod
    def _get_routes(station):
        station = station
        route = 0
        routes = []
        for i in range(station.shape[0]):
            r = station.iloc[i]['ROUTE']
            if route != r:
                routes.append(i)
                route = r
        routes.append(station.shape[0])
        return routes

    def _plot_routes(self, length, scale=1):
        length =  * scale - 0.5
        for route in self.routes:
            route = route * scale - 0.5
            plt.plot([-0.5, length], [route, route], linewidth=0.5, 'b')
            plt.plot([route, route], [-0.5, length], linewidth=0.5, 'b')

    def show_od(self, od):
        sns.heatmap(od, linewidths=0.5)
        self._plot_routes(len(od))



def tickTimes(args, length, axis='x'):
    hour = 60 // args.resolution
    num_hour = args.num_time // hour
    ticks = np.arange(length // hour).astype(int)
    labels = list(map(lambda x: str(x) + ':00', ticks))
    ticks *= hour
    if axis is 'x':
        plt.xticks(ticks, labels, rotation=90)
    else:
        plt.yticks(ticks, labels)
