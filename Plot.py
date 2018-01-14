import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import Data


def saveclf(figpath):
    dirname = os.path.dirname(figpath)
    if not os.path.exists(dirname):
        os.makedirs(os.path.dirname(figpath))
    plt.savefig(figpath + '.png')
    plt.clf()


def plot_network():
    station = Data.load_station(clean=False)
    link = Data.load_link('LINK_RAW.txt')
    plt.axis('off')
    for i in range(link.shape[0]):
        s, e = link[i, 0], link[i, 1]
        if s in station.index and e in station.index:
            plt.plot(station.loc[[s, e], 'Longitude'],
                     station.loc[[s, e], 'Latitude'], 'gray')

def scatter_network(val, indices=None, scale=1, c=None, edgecolors='none'):
    station = Data.load_station()
    if indices is not None:
        assert len(val) == len(indices)
        station = station.iloc[indices]
    else:
        assert len(val) == len(station)
    plt.scatter(station['Longitude'], station['Latitude'],
                s=val*scale, c=c, alpha=0.5, edgecolors=edgecolors)

def scatter_od(indice, val1, val2, c):
    station = Data.load_station()
    x, y = station.iloc[indice]['Longitude'], station.iloc[indice]['Latitude']
    scale = 10 * len(val1) / val1.sum()
    plot_network()
    plt.scatter(x, y, c='red')
    scatter_network(val1, scale=scale, c=c)
    scatter_network(val2, scale=scale, c='none', edgecolors=c)


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
