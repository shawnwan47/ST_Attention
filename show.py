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


def load_atts():
    atts = {}
    for attention in os.listdir(MODEL_PATH):
        modelname = attention.split('_')[0]
        atts[modelname] = np.load(MODEL_PATH + attention)
    return atts


def show_atts_loc2time(atts):
    for modelname, att in atts.items():
        days, times, layers, heads, future, locs, past, locs = att.shape
        im = att.mean(0)
        for lay in range(att.shape[1]):
            for head in range(att.shape[2]):
                for loc in range(att.shape[-1]):
                    Visual.loc2time(im[:, lay, head, 0, loc, 0, :], loc, args)
                    figpath = os.path.join(FIG_PATH, modelname, 'lay' + str(lay),
                                           'head' + str(head), 'loc2time', str(loc))
                    Visual.saveclf(figpath)


def show_atts_mean(atts):
    for modelname, att in atts.items():
        days, times, layers, heads, future, locs, past, locs = att.shape
        im = att.mean(0).mean(0)
        for lay in range(layers):
            for head in range(heads):
                im = att[:, lay, head, 0, :, :, :].mean(0)
                im = im.transpose(0, 2, 1).reshape((locs, -1))
                Visual.loc2loc(im, args)
                figpath = os.path.join(FIG_PATH, modelname, 'lay' + str(lay),
                                    'head' + str(head), 'mean')
                Visual.saveclf(figpath)
                for past in range(pasts):
                    im = att[:, lay, head, 0, :, past, :].mean(0)
                    Visual.loc2loc(im, args)
                    figpath = os.path.join(FIG_PATH, modelname, 'lay' + str(lay),
                                           'head' + str(head), str(past))
                    Visual.saveclf(figpath)


def show_atts_network(atts)


plt.clf()
reload(Visual)

idx = Data.load_idx()
flow = Data.load_flow().loc[:, idx].mean(0)
Visual.plot_network()
Visual.scatter_network(flow)

atts = load_atts()
show_atts_loc2time(atts)
# show_atts_mean(atts)
