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


def loc2time(im, loc, args):
    plt.imshow(im, vmin=0, vmax=0.1)
    tickTimes(args, im.shape[0], 'y')
    plt.xticks([loc], ['location'])
