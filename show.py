import os
import argparse
from imp import reload

import numpy as np
from matplotlib import pyplot as plt

import Args
import Data
import Plot
from Consts import MODEL_PATH, FIG_PATH


plt.rcParams['figure.dpi'] = 600


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


def get_routes():
    station = Data.load_station()
    route = ''
    ret = []
    for i in range(station.shape[0]):
        r = station.iloc[i]['ROUTE']
        if route != r:
            ret.append(i)
            route = r
    ret.append(station.shape[0])
    return ret


def get_routes_od(od, r, by='od'):
    assert by in ['od', 'o', 'd']
    length = len(r) - 1
    for i in range(length):
        for j in range(length):
            s1, e1, s2, e2 = r[i], r[i+1], r[j], r[j+1]
            if by == 'od':
                scale = e1 - s1 + e2 - s2
            elif by == 'o':
                scale = e1 - s1
            else:
                scale = e2 - s2
            val = od[s1:e1, s2:e2].sum() / scale
            od[s1:e1, s2:e2] = val
    return od


def scatter_flow(od='O'):
    assert od in ['O', 'D']
    Plot.plot_network()
    Plot.scatter_network(Data.load_flow(od).mean(0))
    Plot.saveclf(FIG_PATH + od)


def imshow_od(by='od', diagsum=True, routes_od=False):
    assert by in ['o', 'd', 'od']
    if routes_od:
        diagsum = False
    figpath = FIG_PATH + 'imshow_od/' + by
    figpath += '_diag' if diagsum else ''
    figpath += '_routes' if routes_od else ''
    od = Data.load_od()
    if by == 'od':
        scale = od.sum(0) + od.sum(1)
        od = np.log(od + od.mean())
    else:
        if by == 'o':
            scale = od.sum(1, keepdims=True)
        else:
            scale = od.sum(0, keepdims=True)
        od = od / (scale + 1)

    routes = get_routes()
    if routes_od:
        od = get_routes_od(od, routes)
    Plot.imshow_square(od, cmap='gray')
    if diagsum:
        diag = make_diag(by)
        Plot.imshow_square(diag, cmap='cool')
    # plot line splitting routes
    length = len(od) - 0.5
    for route in routes:
        route -= 0.5
        plt.plot([-0.5, length], [route, route], 'red', linewidth=0.3)
        plt.plot([route, route], [-0.5, length], 'red', linewidth=0.3)
    Plot.saveclf(figpath)


def scatter_od(od, indices, by='o'):
    assert by in ['o', 'd']
    odo = od / (od.sum(1, keepdims=True) + 1)
    odd = od / (od.sum(0, keepdims=True) + 1)
    if by == 'o':
        od1, od2 = odo, odd
    else:
        od1, od2 = odd, odo
    for i in indices:
        name = station.iloc[i]['NAME']
        name += 'od' if by == 'o' else 'do'
        x, y = station.iloc[i]['LON'], station.iloc[i]['LAT']
        od1i, od2i = (od1[i], od2[i]) if by == 'o' else (od1[:, i], od2[:, i])
        od2i[od1i < 0.01] = 0
        Plot.plot_network()
        plt.scatter(x, y, c='red')
        Plot.scatter_od(i, od1i, c=od2i, cmap='cool', vmin=0, vmax=1)
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


orig = Data.load_flow('O')
dest = Data.load_flow('D')
orig_max = np.argsort(-orig.mean(0))[:10]
dest_max = np.argsort(-dest.mean(0))[:10]
od = Data.load_od()
