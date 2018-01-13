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
    station = Data.load_station()
    link = Data.load_link('LINK_RAW.txt')
    plt.axis('off')
    for i in range(link.shape[0]):
        s, e = link[i, 0], link[i, 1]
        if s in station.index and e in station.index:
            plt.plot(station.loc[[s, e], 'Longitude'],
                     station.loc[[s, e], 'Latitude'], 'gray')

def scatter_network(val, indices=None, s=1, c=None, label=None):
    station = Data.load_station(clean=True)
    if indices is not None:
        assert len(indices) == len(val)
        station = station.iloc[indices]
    else:
        assert len(val) == len(station)
    plt.scatter(station['Longitude'], station['Latitude'],
                s=val * s, c=c, label=label, alpha=0.5, edgecolors='none')


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


def loc2loc(im, args):
    plt.axis('off')
    plt.imshow(im, vmin=0, vmax=0.1)


def loc2time(im, loc, args):
    plt.imshow(im, vmin=0, vmax=0.1)
    tickTimes(args, im.shape[0], 'y')
    plt.xticks([loc], ['location'])
