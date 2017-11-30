from collections import Counter
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from Constants import FIG_PATH


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def var_avg(flow):
    # variation to historical average
    hist_len = flow.shape[0] // 22 * 14
    flow_hist = flow.iloc[:hist_len]
    flow_futr = flow.iloc[hist_len:]
    day = flow_hist.index.dayofweek
    time = flow_hist.index.time
    daytime = list(zip(day, time))
    hist_avg = dict()
    for dt in Counter(daytime).keys():
        idx = (day == dt[0]) & (time == dt[1])
        hist_avg[dt] = flow_hist[idx].mean().tolist()
    flow_pred = []
    for dt in list(zip(flow_futr.index.dayofweek, flow_futr.index.time)):
        flow_pred.append(hist_avg[dt])
    flow_pred = np.asarray(flow_pred)
    loss = (flow_futr - flow_pred).abs().sum(1) / flow_futr.sum(1)
    loss = loss.as_matrix()
    return loss


def var_prv(flow):
    time = flow.index.time
    loss = (flow.iloc[1:].as_matrix() - flow.iloc[:-1].as_matrix())
    loss = np.abs(loss).sum(1) / flow.iloc[1:].sum(1).as_matrix()
    loss = loss[time[1:] != time[0]]
    return loss


def show_attns(attns, granularity, folderpath='attns/'):
    fig_path = FIG_PATH + folderpath
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    ylen, xlen = attns[0].shape
    assert(60 % granularity == 0)
    step = 60 // granularity

    for i in range(attns.shape[0]):
        def int2time(x):
            return str(int(x)) + ':00'
        plt.clf()
        xstart = min(xlen, 6 * step + i) // step  # forget yesterday
        xticks = np.arange(0, xlen, step)[-xstart:] + 3.5 - i % step
        yticks = np.arange(0, ylen, step) + 3.5 - i % step
        xlabels = 6 + (xticks + i - xlen) // step
        ylabels = 6 + (yticks + i) // step
        xlabels = list(map(int2time, xlabels))
        ylabels = list(map(int2time, ylabels))
        plt.imshow(attns[i], cmap='gray', vmin=0, vmax=1)
        plt.xticks(xticks, xlabels, rotation=45)
        plt.yticks(yticks, ylabels)
        plt.xlabel('Past')
        plt.ylabel('Future')
        title = str(6 + i // step) + ' - ' + str(i % step * granularity)
        plt.title(title)
        plt.savefig(fig_path + title + '.png')


if __name__ == '__main__':
    attns = np.load('attns.npy')
    show_attns(attns, 15)
