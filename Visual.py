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

def scatter_network(val, scale=1):
    idx = Data.load_idx()
    assert len(val) == len(idx)
    station = Data.load_station().loc[idx]
    plt.scatter(station['Longitude'], station['Latitude'],
                s=val * scale, alpha=0.5, edgecolors='none')


def plotTimeTicks(args, length, axis='x'):
    hour = 60 // args.resolution
    num_hour = args.num_time // hour
    ticks = np.arange(length // hour).astype(int)
    labels = list(map(lambda x: str(x) + ':00', ticks))
    ticks *= hour
    if axis is 'x':
        plt.xticks(ticks, labels, rotation=90)
    else:
        plt.yticks(ticks, labels)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def plot_flow_real(flow, flow_max, args):
    plt.clf()
    plt.plot(flow, label='Real', color='k', linewidth=3)
    length = flow.shape[0]
    plotTimeTicks(args, length, axis='x', offset=True)
    plt.ylim(0, num2ceil(flow_max))


def plot_flow_steps(flows):
    xlen, steps = flows.shape
    x = np.arange(xlen)
    for step in range(steps):
        plt.plot(x + step, flows[:, step], label=str(step) + '-step')



def num2ceil(num):
    num = str(int(num))
    digits = len(num)
    cap = int(num[0]) + 1
    return cap * 10 ** (digits - 1)


def plot_errs(errors):
    for k, l in errors.items():
        length = len(l)
        plt.plot(range(1, 1 + length), l, label=k)
    plt.ylim(0, 1)
    plt.ylabel('Error')
    plt.xlabel('Step')
    plt.legend()


def show_attn(attn, args):
    plt.clf()
    attn_copy = attn.copy()
    h, w = attn_copy.shape
    mask_u = np.triu(np.ones_like(attn_copy), w - h) > 0
    mask_l = np.tril(np.ones_like(attn_copy), w - h - 96) > 0
    attn_copy[mask_u] = np.nan
    attn_copy[mask_l] = np.nan
    plt.imshow(attn_copy, cmap='rainbow', vmin=0, vmax=1)
    ylen, xlen = attn_copy.shape
    plotTimeTicks(args, xlen, axis='x', offset=False)
    plotTimeTicks(args, ylen, axis='y', offset=True)
    plt.xlabel('Past')
    plt.ylabel('Future')


def loc2loc(im, args):
    plt.axis('off')
    plt.imshow(im, vmin=0, vmax=0.1)
    dim = args.num_loc
    mid = dim // 2
    plt.plot(mid * np.ones(dim), 'k')
    plt.plot(mid * np.ones(dim), np.arange(dim), 'k')


def loc2time(im, loc, args):
    plt.imshow(im, vmin=0, vmax=0.1)
    plotTimeTicks(args, im.shape[0], 'y')
    plt.xticks([loc], ['location'])
