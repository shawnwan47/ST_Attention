from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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


if __name__ == '__main__':
    pass
