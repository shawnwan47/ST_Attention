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


def scatter_flow(od='OD'):
    assert od in ['O', 'D', 'OD']
    Plot.plot_network()
    for flow in od:
        Plot.scatter_network(Data.load_flow(flow).mean(0), label=flow)
    plt.legend()
    Plot.saveclf(FIG_PATH + flow)


def imshow_od(norm='od', diagsum=False):
    figpath = FIG_PATH + 'OD_' + norm
    figpath += '_diag' if diagsum else ''
    od = Data.load_od()
    diag = np.diag_indices_from(od)
    if norm is 'od':
        if diagsum:
            scale = od.sum(0) + od.sum(1)
            od[diag] = scale
        od = np.log(od + 1)
    else:
        elif norm == 'o':
            scale = od.sum(1, keepdims=True)
            od = od / scale
        elif norm == 'd':
            scale = od.sum(0, keepdims=True)
            od = od / scale
        if diagsum:
            od[diag] = scale / scale.sum()

    Plot.loc2loc(od, args)
    Plot.saveclf(figpath)


def scatter_od(indices, by='o'):
    od = Data.load_od()
    if by == 'o':
        od = od / (od.sum(1) + 1)
    else:
        od = od / (od.sum(0) + 1)
    station = Data.load_station()
    for i in indices:
        name = station.iloc[i]['Name']
        name += 'od' if by == 'o' else 'do'
        x, y = station.iloc[i]['Longitude'], station.iloc[i]['Latitude']
        od_i = od[i] if by == 'o' else od[:, i]
        scale = 10 * args.num_loc
        Plot.plot_network()
        plt.scatter(x, y, c='red')
        Plot.scatter_network(od_i, s=scale)
        Plot.saveclf(FIG_PATH + name)


def att_loc2loc():
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, _, future, locs, pasts, locs = att.shape
        att = att.mean(0).mean(0)
        for lay in range(layers):
            im = att[lay, 0, 0, :, :, :].sum(1)
            Plot.loc2loc(im, args)
            figpath = os.path.join(FIG_PATH, modelname,
                                   'lay' + str(lay),
                                   'att_loc2loc')
            Plot.saveclf(figpath)


def att_loc2time(selected=None):
    station = Data.load_station()
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        im = att.mean(0)
        for lay in range(layers):
            for loc in selected:
                name = station.iloc[loc]['Name']
                Plot.loc2time(im[:, lay, 0, 0, loc, :, :].sum(1), loc, args)
                figpath = os.path.join(FIG_PATH, modelname,
                                       'lay' + str(lay),
                                       'att_loc2time',
                                       name)
                Plot.saveclf(figpath)
                for past in range(pasts):
                    Plot.loc2time(im[:, lay, 0, 0, loc, past, :], loc, args)
                    figpath = os.path.join(FIG_PATH, modelname,
                                           'lay' + str(lay),
                                           'att_loc2time',
                                           name + str(past))
                    Plot.saveclf(figpath)


def att_scatter(selected=None):
    station = Data.load_station()
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        att = att.mean(0).mean(0)
        for lay in range(layers):
            for loc in selected:
                name = station.iloc[loc]['Name']
                x, y = station.iloc[loc]['Longitude'], station.iloc[loc]['Latitude']
                objs = att[lay, 0, 0, loc, :, :].mean(0)
                Plot.plot_network()
                Plot.scatter_network(objs, 100 * locs)
                plt.scatter(x, y, c='red')
                figpath = os.path.join(FIG_PATH, modelname,
                                       'lay' + str(lay),
                                       'att_scatter', name)
                Plot.saveclf(figpath)


station = Data.load_station(True)
orig = Data.load_flow('O')
dest = Data.load_flow('D')
orig_max = np.argsort(-orig.mean(0))[:10]
dest_max = np.argsort(-dest.mean(0))[:10]
