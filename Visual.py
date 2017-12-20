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
    hour = 60 // args.resolution
    daily_hour = args.daily_times // hour
    ticks = np.arange(length // hour + 1).astype(int)
    labels = list(map(lambda x: str(x) + ':00',
                      ticks % daily_hour + args.start_time))
    ticks *= hour
    if axis is 'x':
        plt.xticks(ticks, labels, rotation=60)
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
    plt.imshow(attn_copy, cmap='rainbow')
    ylen, xlen = attn_copy.shape
    plotTimeTicks(args, xlen, axis='x', offset=False)
    plotTimeTicks(args, ylen, axis='y', offset=True)
    plt.xlabel('Past')
    plt.ylabel('Future')


def show_attns(attns, args, day=7):
    attn_merge, attn_channel, attn_context = attns
    attn_merge = attn_merge[day]
    attn_channel = attn_channel[day]
    attn_context = attn_context[day]


def show_attn_merge(attn_merge, args):
    '''
    attn_merge: length x channel
    '''
    length, channel = attn_merge.shape
    plt.imshow(attn_merge, cmap='rainbow')
    plotTimeTicks(args, length, axis='y')


def chain_attns(attn1, attn2):
    '''
    attn1: len1 x len2
    attn2: len2 x len3
    ret: len1 x len3
    '''
    assert attn1.shape[1] == attn2.shape[0]
    return np.matmul(attn1, attn2)


def show_attn_channel(attn_channel, a):
    pass


