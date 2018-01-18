import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import Data


plt.rcParams['figure.dpi'] = 600


def saveclf(figpath):
    dirname = os.path.dirname(figpath)
    if not os.path.exists(dirname):
        os.makedirs(os.path.dirname(figpath))
    plt.savefig(figpath + '.png', transparent=True)
    plt.clf()

def get_routes():
    station = Data.load_station()
    route = ''
    ret = []
    for i in range(station.shape[0]):
        r = station.iloc[i]['ROUTE']
        if route != r:
            ret.append(i)
            route = r
    ret.append(station.shape[0])
    return ret

def get_od_routes(od, by='od'):
    assert by in ['od', 'o', 'd']
    r = get_routes()
    length = len(r) - 1
    for i in range(length):
        for j in range(length):
            s1, e1, s2, e2 = r[i], r[i+1], r[j], r[j+1]
            if by == 'o':
                scale = e1 - s1
            elif by == 'd':
                scale = e2 - s2
            else:
                scale = e1 - s1 + e2 - s2
            val = od[s1:e1, s2:e2].sum() / scale
            od[s1:e1, s2:e2] = val
    return od



def plot_network():
    station = Data.load_station(clean=False)
    link = Data.load_link(raw=True)
    plt.axis('off')
    for i in range(link.shape[0]):
        s, e = link[i, 0], link[i, 1]
        if s in station.index and e in station.index:
            plt.plot(station.loc[[s, e], 'LON'],
                     station.loc[[s, e], 'LAT'],
                     color='gray', linewidth=1)

def scatter_network(val, indices=None, scale=1, **args):
    station = Data.load_station()
    if indices is not None:
        assert len(val) == len(indices)
        station = station.iloc[indices]
    else:
        assert len(val) == len(station)
    plt.scatter(station['LON'], station['LAT'],
                s=val*scale, alpha=0.5, edgecolors='none', **args)


def scatter_od(indice, val1, **args):
    station = Data.load_station().iloc[indice]
    x, y = station['LON'], station['LAT']
    scale = 10 * len(val1) / val1.sum()
    plt.scatter(x, y, c='black', s=3)
    scatter_network(val1, scale=scale, **args)
    plot_network()


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


def imshow_square(im, **args):
    plt.axis('off')
    plt.imshow(im, **args)


def plot_routes(scale=1, **args):
    length = len(Data.load_idx()) * scale - 0.5
    for route in get_routes():
        route = route * scale - 0.5
        plt.plot([-0.5, length], [route, route], linewidth=0.3, **args)
        plt.plot([route, route], [-0.5, length], linewidth=0.3, **args)


def loc2time(im, loc, args):
    plt.imshow(im, vmin=0, vmax=0.1)
    tickTimes(args, im.shape[0], 'y')
    plt.xticks([loc], ['location'])
