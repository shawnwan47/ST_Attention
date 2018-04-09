'''
Data visualization of Spatial Attention Model for traffic flow prediction.

The main figs are:

- Map of traffic stations (fine)
- Traffic continuity, periodicity(tidal, daily, weekly) to frequencies
- OD relations
- attention visualization to reveal traffic dynamics
    - validate attention with real OD distributions
    - multi-head attention
- traffic flow prediction results
    - attention attributes and prediction results
    - different steps prediction uncertainties
- visualizing t-sne of embeddings of day, time, loc
'''
from imp import reload
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import Args
from Data import Loader
import Plot


# plt.style.use('seaborn')
# plt.rcParams['figure.dpi'] = 600

plt.clf()
reload(Plot)
reload(Data)


DATASET = 'highway'
FIG_PATH = Path('./fig/') / DATASET

def make_diag(by='od'):
    assert by in ['o', 'd', 'od']
    od = Data.load_od()
    if by is 'o':
        scale = od.sum(1)
    elif by is 'd':
        scale = od.sum(0)
    else:
        scale = od.sum(0) + od.sum(1)
    nan = np.zeros_like(od) * np.nan
    nan[np.diag_indices_from(od)] = np.log(scale + scale.mean())
    return nan


def get_od_expand(od, pos=0, r=2):
    assert pos in list(range(r))
    od_ = np.zeros((len(od) * r, len(od) * r)) * np.nan
    od_[pos::r, pos::r] = od[:]
    return od_


def scatter_flow(od='O', max_count=10, scale=1):
    assert od in ['O', 'D']
    flow = Data.load_flow(od).mean(0).as_matrix()
    Plot.plot_network()
    Plot.scatter_network(flow, scale=scale)
    Plot.saveclf(FIG_PATH + od)
    rank = np.argsort(-flow)
    rankmax = rank[:max_count]
    rankres = rank[max_count:]
    Plot.plot_network()
    Plot.scatter_network(flow[rankres], rankres, scale=scale)
    Plot.scatter_network(flow[rankmax], rankmax, scale=scale)
    Plot.saveclf(FIG_PATH + od + '_max')


def imshow_od(by='od', diagsum=True, routes_od=False):
    assert by in ['o', 'd', 'od']
    if routes_od:
        diagsum = False
    figpath = FIG_PATH + 'imshow_od/' + by
    figpath += '_diag' if diagsum else ''
    figpath += '_routes' if routes_od else ''
    od = Data.load_od()
    if by == 'od':
        scale = od.sum(0) + od.sum(1) + 1
        od = np.log(od + od.mean())
    else:
        if by == 'o':
            scale = od.sum(1, keepdims=True)
        else:
            scale = od.sum(0, keepdims=True)
        od = od / (scale + 1)

    if routes_od:
        od = Plot.get_od_routes(od)
    Plot.imshow_square(od, cmap='gray')
    if diagsum:
        diag = make_diag(by)
        Plot.imshow_square(diag, cmap='cool')
    # plot line splitting routes
    Plot.plot_routes(color='red')
    Plot.saveclf(figpath)


def imshow_od_mean(routes_od=False, cmap='gray'):
    assert cmap in ['gray', 'cool']
    figpath = FIG_PATH + 'imshow_od/od_mean'
    figpath += '_routes' if routes_od else ''
    if cmap == 'gray':
        color = 'red'
    if cmap == 'cool':
        color = 'black'
    o = Data.load_od()
    d = Data.load_od()
    o[:] = o.sum(1, keepdims=True)
    d[:] = d.sum(0)
    if routes_od:
        o = Plot.get_od_routes(o)
        d = Plot.get_od_routes(d)
    # o = np.log(o + o.mean())
    # d = np.log(d + d.mean())
    o = get_od_expand(o, pos=0)
    d = get_od_expand(d, pos=1)
    Plot.imshow_square(o, cmap=cmap)
    Plot.imshow_square(d, cmap=cmap)
    Plot.plot_routes(scale=2, color=color)
    Plot.saveclf(figpath)


def scatter_od(by='o', count=10, vmax=1):
    assert by in ['o', 'd']
    od = Data.load_od()
    station = Data.load_station()
    odo = od / (od.sum(1, keepdims=True) + 1)
    odd = od / (od.sum(0, keepdims=True) + 1)
    if by == 'o':
        od1, od2 = odo, odd
        indices = np.argsort(-od.mean(1))[:count]
    else:
        od1, od2 = odd, odo
        indices = np.argsort(-od.mean(0))[:count]
    for i in indices:
        name = station.iloc[i]['NAME']
        name += 'od' if by == 'o' else 'do'
        x, y = station.iloc[i]['LON'], station.iloc[i]['LAT']
        od1i, od2i = (od1[i], od2[i]) if by == 'o' else (od1[:, i], od2[:, i])
        od2i[od1i < 0.01] = 0
        Plot.plot_network()
        plt.scatter(x, y, c='red')
        Plot.scatter_od(i, od1i, c=od2i, cmap='cool', vmin=0, vmax=vmax)
        Plot.saveclf(FIG_PATH + 'scatter_od/' + name)


def imshow_att():
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, _, future, locs, pasts, locs = att.shape
        att_am = att.mean(0)[4*6:4*10].mean(0)
        att_pm = att.mean(0)[4*16:4*20].mean(0)
        for lay in range(layers):
            im_am = att_am[lay, 0, 0].sum(1)
            im_pm = att_pm[lay, 0, 0].sum(1)
            im = (im_am + im_pm) / 2
            Plot.imshow_square(im, args)
            figpath = os.path.join(FIG_PATH, modelname, 'lay' + str(lay), 'im')
            Plot.saveclf(figpath)
            Plot.imshow_square(im_am, args)
            figpath = os.path.join(FIG_PATH, modelname, 'lay' + str(lay), 'im_am')
            Plot.saveclf(figpath)
            Plot.imshow_square(im_pm, args)
            figpath = os.path.join(FIG_PATH, modelname, 'lay' + str(lay), 'im_pm')
            Plot.saveclf(figpath)


def att_loc2time(indices=None):
    station = Data.load_station()
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        im = att.mean(0)
        for lay in range(layers):
            for i in indices:
                name = station.iloc[i]['NAME']
                Plot.loc2time(im[:, lay, 0, 0, i].sum(1), i, args)
                figpath = os.path.join(FIG_PATH, modelname,
                                       'lay' + str(lay),
                                       'att_loc2time',
                                       name)
                Plot.saveclf(figpath)
                for past in range(pasts):
                    Plot.loc2time(im[:, lay, 0, 0, i, past, :], i, args)
                    figpath = os.path.join(FIG_PATH, modelname,
                                           'lay' + str(lay),
                                           'att_loc2time',
                                           name + str(past))
                    Plot.saveclf(figpath)


def scatter_att(indices=None):
    station = Data.load_station()
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        att = att.mean(0).mean(0)
        for lay in range(layers):
            for i in indices:
                name = station.iloc[i]['NAME']
                x, y = station.iloc[i]['LON'], station.iloc[i]['LAT']
                im = att[lay, 0, 0, i].mean(0)
                Plot.plot_network()
                Plot.scatter_network(im, 100 * locs)
                plt.scatter(x, y, c='red')
                figpath = os.path.join(FIG_PATH, modelname,
                                       'lay' + str(lay),
                                       'scatter_att', name)
                Plot.saveclf(figpath)


def windowLegend(window):
    minute = window * 5
    if minute < 60:
        return str(minute) + 'm'
    else:
        assert not minute % 60
        return str(minute // 60) + 'h'


def plot_rolling_diff(flow, func='cont'):
    def rolling_diff(flow, window=1, periods=1):
        assert flow.index.freqstr == '5T'
        flow = flow.rolling(window).mean()
        ret = pd.DataFrame(index=flow.index)
        ret['hour'] = ret.index.map(lambda x: x.hour * 60 + x.minute)
        ret['unit'] = ret.index.map(lambda x: x.dayofyear)
        ret['Error'] = flow.diff(periods).abs().sum(1).div(flow.sum(1))
        ret['Freq'] = windowLegend(window)

        ret = ret[(ret['hour'] >= 60 * 6) & (ret['hour'] <= 60 * 22)]
        weekday = ret.index.map(lambda x: x.weekday)
        ret = ret[(weekday < 5) & (weekday > 0)]
        return ret

    windows = [1, 2, 3, 4, 6, 12, 24]
    flow = pd.concat([rolling_diff(flow, window, window if func=='cont' else 288)
                      for window in windows])
    ax = sns.tsplot(data=flow, time="hour", unit='unit', value="Error", condition="Freq", ci='sd')
    ax.set_xticks(np.arange(6, 23) * 60)
    ax.set_xticklabels(np.arange(6, 23))
    return ax


def plot_traffic_cont_peri(flow, data_string):
    ax_o = plot_rolling_diff(flow, func='cont')
    ax_o.set_title("Traffic Continuity of " + data_string)
    plt.tight_layout()
    plt.savefig(FIG_PATH / (data_string + '_continuity.png'))
    plt.show()

    ax_o = plot_rolling_diff(flow, func='peri')
    ax_o.set_title("Traffic Periodicity of " + data_string)
    plt.tight_layout()
    plt.savefig(FIG_PATH / (data_string + '_periodicity.png'))
    plt.show()


if __name__ == '__main__':
    loader = Data.Loader(DATASET)
    station_raw = loader.load_station_raw()
    station = loader.load_station()
    O = loader.load_flow('O')
    D = loader.load_flow('D')
    OD = loader.load_od('OD')
    DO = loader.load_od('DO')

    plot_traffic_cont_peri(O, 'Entries')
    plot_traffic_cont_peri(D, 'Exits')
    plot_traffic_cont_peri(OD.unstack([-1, -2], fill_value=0), 'OD')
    plot_traffic_cont_peri(DO.unstack([-1, -2], fill_value=0), 'DO')
