import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def saveclf(figpath):
    dirname = os.path.dirname(figpath)
    if not os.path.exists(dirname):
        os.makedirs(os.path.dirname(figpath))
    plt.savefig(figpath + '.png')
    plt.clf()


def plotTimeTicks(args, length, axis='x', offset=True):
    hour = 60 // args.gran
    if offset:
        start_time = args.start_time + args.past // hour
    else:
        start_time = args.start_time
    ticks = np.arange(length // hour + 1).astype(int)
    labels = list(map(lambda x: str(x) + ':00',
                      ticks % (args.end_time - args.start_time) + start_time))
    ticks *= hour
    if axis is 'x':
        plt.xticks(ticks, labels, rotation=45)
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


def show_attn(attn, args):
    plt.clf()
    h, w = attn.shape
    mask = np.triu(np.ones_like(attn), w - h) > 0
    attn[mask] = np.nan
    plt.imshow(attn)
    ylen, xlen = attn.shape
    plotTimeTicks(args, xlen, axis='x', offset=False)
    plotTimeTicks(args, ylen, axis='y', offset=True)
    plt.xlabel('Past')
    plt.ylabel('Future')


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
