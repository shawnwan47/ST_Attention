from os import path
import argparse
from imp import reload

import numpy as np
from matplotlib import pyplot as plt

import Args
import Visual
from Consts import MODEL_PATH, FIG_PATH


# LOAD DATA
args = argparse.ArgumentParser()
Args.add_gpu(args)
Args.add_data(args)
Args.add_loss(args)
Args.add_optim(args)
Args.add_run(args)
Args.add_model(args)
args = args.parse_args()
Args.update_args(args)


def get_outs(model):
    ret = []
    for i in range(10):
        try:
            ret.append(np.load(
                MODEL_PATH + model + '_out_' + str(i) + '.npy'))
        except FileNotFoundError:
            return ret


def dict_name(errs):
    name = ''
    for key in errs.keys():
        name += key
    return name


def plot_attn_head_time_day(attn, modelname):
    head, time, day, length, _ = attn.shape
    for d in range(1):
        for h in range(head):
            for t in range(time):
                att = attn[h, t, d]
                plt.imshow(att, cmap='rainbow')
                plt.axis('off')
                Visual.saveclf(path.join(
                    FIG_PATH, modelname,
                    'day' + str(d),
                    'head' + str(h),
                    str(t)))

def plot_attn_head_day(attn, modelname):
    head, day, _, _ = attn.shape
    for d in range(day):
        for h in range(head):
            att = attn[h, d]
            plt.imshow(att, cmap='rainbow')
            plt.axis('off')
            Visual.saveclf(path.join(FIG_PATH, modelname, str(h), str(d)))


def plot_dilated():
    attns = {'Attn'        : get_outs('STAttnChan1Lay1Hid1024Day16Time64Past1Future4'),
             'AttnDilemb'  : get_outs('STAttnDilatedChan1Lay3Hid1024Day16Time64Past1Future4'),
             'AttnDilC4emb': get_outs('STAttnDilatedChan4Lay3Hid1024Day16Time64Past1Future4'),
             'AttnDilC4'   : get_outs('STAttnDilatedChan4Lay3Hid1024Past1Future4')}

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


if __name__ == '__main__':
    plt.clf()
    reload(Visual)
    modelname = 'EnTempAttnHead4Head4addLay1TimeFuture4'
    out = get_outs(modelname)
    attn, weight = out
    weight = weight.squeeze()  # model x day x head x time x time
    # plot_attn_head_day(out[0], temporal)
    # plot_attn_head_time_day(out[0], modelname)
