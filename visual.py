from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

import data_utils


FIG_PATH = './fig/'


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
    for flowname in ['OD', 'DO']:
        flow = data_utils.load_data(flowname)
        for fun, varname in zip([var_avg, var_prv], ['avg', 'prv']):
            for gran in [5, 10, 15, 20, 30, 60]:
                flow_tmp = data_utils.resample_data(flow, gran)
                loss = fun(flow_tmp)
                idx = flow_tmp.index
                if fun == var_avg:
                    hist_len = flow_tmp.shape[0] // 22 * 14
                    idx = idx[hist_len:]
                if fun == var_prv:
                    idx = idx[idx.time != idx.time[0]]
                flow_tmp = flow_tmp.loc[idx]
                loss_mean = (loss * flow_tmp.sum(1)).sum()
                loss_mean /= flow_tmp.sum().sum()

                if fun == var_avg:
                    loss = loss.reshape(8, -1).mean(0)
                if fun == var_prv:
                    loss = loss.reshape(22, -1).mean(0)

                plt.plot_date(
                    idx.time[:loss.shape[0]],
                    loss,
                    '-',
                    label=str(gran) + ': ' + str(loss_mean))
            plt.axis('tight')
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(FIG_PATH + flowname + varname + '.png')
            plt.clf()
