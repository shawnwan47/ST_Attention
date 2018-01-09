import os
import argparse
from imp import reload

import numpy as np
from matplotlib import pyplot as plt

import Args
import Data
import Visual
from Consts import MODEL_PATH, FIG_PATH


# LOAD DATA
args = argparse.ArgumentParser()
Args.add_data(args)
Args.add_model(args)
args = args.parse_args()
Args.update_args(args)


def att_loc2time():
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        im = att.mean(0)
        for lay in range(layers):
            for head in range(heads):
                for loc in range(locs):
                    Visual.loc2time(im[:, lay, head, 0, loc, 0, :], loc, args)
                    figpath = os.path.join(FIG_PATH, modelname,
                                           'lay' + str(lay),
                                           'head' + str(head),
                                           'loc2time',
                                           str(loc))
                    Visual.saveclf(figpath)


def att_loc2loc():
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        att = att.mean(0).mean(0)
        for lay in range(layers):
            for head in range(heads):
                for past in range(pasts):
                    im = att[lay, head, 0, :, past, :]
                    Visual.loc2loc(im, args)
                    figpath = os.path.join(FIG_PATH, modelname,
                                           'lay' + str(lay),
                                           'head' + str(head),
                                           str(past))
                    Visual.saveclf(figpath)


def scatter_flow():
    idx = Data.load_idx()
    flow = Data.load_flow().loc[:, idx].mean(0)
    Visual.plot_network()
    Visual.scatter_network(flow)


def att_scatter():
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        att = np.load(MODEL_PATH + attention)
        days, times, layers, heads, future, locs, pasts, locs = att.shape
        att = att.mean(0).mean(0)
        for lay in range(layers):
            for head in range(heads):
                for past in range(pasts):
                    loc = att[lay, head, 0, :, past, :].mean(0)
                    Visual.plot_network()
                    Visual.scatter_network(loc[:locs//2], 100 * locs)
                    Visual.scatter_network(loc[locs//2:], 100 * locs)
                    figpath = os.path.join(FIG_PATH, modelname,
                                           'lay' + str(lay),
                                           'head' + str(head),
                                           'signal')
                    Visual.saveclf(figpath)



plt.clf()
reload(Visual)
