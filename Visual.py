import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from Consts import FIG_PATH, DAYS_TEST
import Data


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def plot_predictions(targets, outputs, args):
    plt.clf()
    times = targets.shape[0]
    times_, steps = outputs.shape
    assert times == times_ + steps
    onehour = 60 // args.gran
    xticks = np.arange(times // onehour)  # compute hours
    start_time = args.start_time + args.past // onehour
    xtick_labels = [str(start_time + i) + ':00' for i in xticks]
    xticks *= steps
    x = np.arange(times)
    plt.plot(x, targets, label='Truth')
    for step in range(steps):
        plt.plot(x[step:-steps + step], outputs[:, step], label=str(step))
    plt.legend()
    plt.xticks(xticks, xtick_labels, rotation=45)


def var_prv(flow):
    time = flow.index.time
    loss = (flow.iloc[1:].as_matrix() - flow.iloc[:-1].as_matrix())
    loss = np.abs(loss).sum(1) / flow.iloc[1:].sum(1).as_matrix()
    loss = loss[time[1:] != time[0]]
    return loss


def show_attns(attns, args):
    plt.clf()
    plt.imshow(attns)

    def len2ticklabels(length, start_time):
        hour = 60 // args.gran
        ticks = np.arange(length // hour).astype(int)
        labels = list(map(lambda x: str(x) + ':00', ticks + start_time))
        ticks *= hour
        return ticks, labels

    ylen, xlen = attns.shape
    xticks, xlabels = len2ticklabels(xlen, args.start_time)
    yticks, ylabels = len2ticklabels(ylen, args.start_time + args.past)
    plt.xticks(xticks, xlabels, rotation=45)
    plt.yticks(yticks, ylabels)
    plt.xlabel('Past')
    plt.ylabel('Prediction')
