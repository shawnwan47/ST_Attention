from os import path
import argparse
from imp import reload

import numpy as np
from matplotlib import pyplot as plt

import Args
import Visual
from Consts import MODEL_PATH, FIG_PATH


# CONFIG
args = argparse.ArgumentParser('Traffic Forecasting')
Args.add_gpu(args)
Args.add_data(args)
Args.add_loss(args)
Args.add_optim(args)
Args.add_run(args)
Args.add_model(args)
args = args.parse_args()
Args.update_args(args)
print(args)

# LOAD DATA


def get_loss(model):
    return np.genfromtxt(MODEL_PATH + model + '_loss.txt')


def get_attn(model):
    return np.load(MODEL_PATH + model + '_attn.npy')


def get_attns(model):
    ret = []
    for i in range(10):
        try:
            ret.append(np.load(
                MODEL_PATH + model + '_attn_' + str(i) + '.npy'))
        except FileNotFoundError:
            return ret


def load_attns():
    attns = {}
    args.model = 'STAttn'
    args.daytime = True
    modelname = Args.modelname(args)
    try:
        attns[modelname] = get_attns(modelname)
    except FileNotFoundError:
        pass
    return attns



# attn
def plot_attn():
    for model, attn in attns.items():
        path_model = FIG_PATH + model + '/'
        attn.shape


def dict_name(errs):
    name = ''
    for key in errs.keys():
        name += key
    return name


if __name__ == '__main__':
    plt.clf()
    reload(Visual)

    attns = {'Attn'        : get_attns('STAttnChan1Lay1Hid1024Day16Time64Past1Future4'),
             'AttnDilemb'  : get_attns('STAttnDilatedChan1Lay3Hid1024Day16Time64Past1Future4'),
             'AttnDilC4emb': get_attns('STAttnDilatedChan4Lay3Hid1024Day16Time64Past1Future4'),
             'AttnDilC4'   : get_attns('STAttnDilatedChan4Lay3Hid1024Past1Future4')}

    for key, val in attns.items():
        path_model = FIG_PATH + key + '/'
        attn_merge, attn_channel, attn_context = val
        for day in range(7):
            path_day = path_model + str(day) + '/'
            for layer in range(attn_merge.shape[1]):
                path_layer = path_day + 'lay' + str(layer)
                merge = attn_merge[day, layer]
                Visual.show_attn_merge(merge, args)
                Visual.saveclf(path_layer + 'merge')
                for c in range(attn_channel.shape[2]):
                    path_channel = path_layer + 'ch' + str(c)
                    channel = attn_channel[day, layer, c]
                    context = attn_context[day, layer, c]
                    Visual.show_attn(channel, args)
                    Visual.saveclf(path_channel + 'channel')
                    Visual.show_attn(context, args)
                    Visual.saveclf(path_channel + 'context')
                    tmp = channel
                    while layer > 0:
                        context = attn_context[day, layer - 1, c]
                        tmp = np.matmul(tmp, context)
                        Visual.show_attn(tmp, args)
                        Visual.saveclf(path_channel + 'chancont' + str(layer))
                        layer -= 1

