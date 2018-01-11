import os
import argparse
from imp import reload

import numpy as np
from matplotlib import pyplot as plt

import Args
import Data
import Visual
from Consts import MODEL_PATH, FIG_PATH


args = argparse.ArgumentParser()
Args.add_args(args)
args = args.parse_args()
Args.update_args(args)
print(args)

plt.clf()
reload(Visual)
reload(Data)


def scatter_flow():
    flow = Data.load_flow(clean=True).mean(0)
    Visual.plot_network()
    Visual.scatter_network(flow)
    Visual.saveclf(FIG_PATH + 'flow')


def scatter_selected(selected, flow):
    Visual.plot_network()
    Visual.scatter_network_highlight(selected, flow, c='red')
    Visual.saveclf(FIG_PATH + 'selected')
    return selected


def show_od():
    od = Data.load_od()
    od = od / (od.sum(0) + 1)
    od = od.transpose()
    Visual.loc2loc(od, args)
    Visual.saveclf(FIG_PATH + 'OD')


def scatter_od(selected):
    od = Data.load_od()
    od = od / (od.sum(0) + 1)
    od = od.transpose()
    station = Data.load_station(clean=True)
    for loc in selected:
        name = station.iloc[loc]['Name']
        x, y = station.iloc[loc]['Longitude'], station.iloc[loc]['Latitude']
        Visual.plot_network()
        Visual.scatter_network(od[loc], 10 * args.num_loc)
        plt.scatter(x, y, c='red')
        Visual.saveclf(FIG_PATH + name + 'od')



def att_loc2loc():
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        att = att.mean(0).mean(0)
        for lay in range(layers):
            im = att[lay, 0, 0, :, :, :].sum(1)
            Visual.loc2loc(im, args)
            figpath = os.path.join(FIG_PATH, modelname,
                                   'lay' + str(lay),
                                   'att_loc2loc')
            Visual.saveclf(figpath)


def att_loc2time(selected=None):
    station = Data.load_station(clean=True)
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        im = att.mean(0)
        for lay in range(layers):
            for loc in selected:
                name = station.iloc[loc]['Name']
                Visual.loc2time(im[:, lay, 0, 0, loc, :, :].sum(1), loc, args)
                figpath = os.path.join(FIG_PATH, modelname,
                                       'lay' + str(lay),
                                       'att_loc2time',
                                       name)
                Visual.saveclf(figpath)
                for past in range(pasts):
                    Visual.loc2time(im[:, lay, 0, 0, loc, past, :], loc, args)
                    figpath = os.path.join(FIG_PATH, modelname,
                                           'lay' + str(lay),
                                           'att_loc2time',
                                           name + str(past))
                    Visual.saveclf(figpath)


def att_scatter(selected=None):
    station = Data.load_station(clean=True)
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
                Visual.plot_network()
                Visual.scatter_network(objs, 100 * locs)
                plt.scatter(x, y, c='red')
                figpath = os.path.join(FIG_PATH, modelname,
                                       'lay' + str(lay),
                                       'att_scatter', name)
                Visual.saveclf(figpath)



flow = Data.load_flow(clean=True).mean(0).as_matrix()
selected = np.argsort(-flow)[:10]
att_loc2loc()
att_loc2time(selected)
