import os
import argparse
from imp import reload

import numpy as np
from matplotlib import pyplot as plt

import Args
import Data
import Plot
from Consts import MODEL_PATH, FIG_PATH


args = argparse.ArgumentParser()
Args.add_args(args)
args = args.parse_args()
Args.update_args(args)
print(args)

plt.clf()
reload(Plot)
reload(Data)


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


def plot_scatter(val):
    pass


def scatter_flow(od='O'):
    assert od in ['O', 'D']
    Plot.plot_network()
    Plot.scatter_network(Data.load_flow(od).mean(0))
    Plot.saveclf(FIG_PATH + od)


def imshow_od(by='od', diagsum=True):
    assert by in ['o', 'd', 'od']
    figpath = FIG_PATH + 'OD_' + by
    figpath += '_diagsum' if diagsum else ''
    od = Data.load_od()
    if by == 'od':
        scale = od.sum(0) + od.sum(1)
        od = np.log(od + od.mean())
    else:
        if by == 'o':
            scale = od.sum(1, keepdims=True)
        elif by == 'd':
            scale = od.sum(0, keepdims=True)
        od = od / (scale + 1)

    Plot.imshow_square(od)
    if diagsum:
        diag = make_diag(by)
        Plot.imshow_square(diag, cmap='jet')
    Plot.saveclf(figpath)


def scatter_od(od, indices, by='o'):
    assert by in ['o', 'd']
    odo = od / (od.sum(1, keepdims=True) + 1)
    odd = od / (od.sum(0, keepdims=True) + 1)
    if by == 'o':
        od1, od2 = odo, odo / (odd + 1e-8)
    else:
        od1, od2 = odd, odd / (odo + 1e-8)
    for i in indices:
        name = station.iloc[i]['Name']
        name += 'od' if by == 'o' else 'do'
        x, y = station.iloc[i]['Longitude'], station.iloc[i]['Latitude']
        od1i, od2i = (od1[i], od2[i]) if by == 'o' else (od1[:, i], od2[:, i])
        od2i[od1i < 0.01] = 0
        Plot.plot_network()
        plt.scatter(x, y, c='red')
        Plot.scatter_od(i, od1i, od2i, c='blue')
        Plot.saveclf(FIG_PATH + name)


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
                name = station.iloc[i]['Name']
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
                name = station.iloc[i]['Name']
                x, y = station.iloc[i]['Longitude'], station.iloc[i]['Latitude']
                im = att[lay, 0, 0, i].mean(0)
                Plot.plot_network()
                Plot.scatter_network(im, 100 * locs)
                plt.scatter(x, y, c='red')
                figpath = os.path.join(FIG_PATH, modelname,
                                       'lay' + str(lay),
                                       'scatter_att', name)
                Plot.saveclf(figpath)


station = Data.load_station(True)
orig = Data.load_flow('O')
dest = Data.load_flow('D')
orig_max = np.argsort(-orig.mean(0))[:10]
dest_max = np.argsort(-dest.mean(0))[:10]
od = Data.load_od()
