from os import path
import argparse
from imp import reload

import numpy as np
from matplotlib import pyplot as plt

import Args
import Visual
from Consts import DAYS_TEST, MODEL_PATH, FIG_PATH


# CONFIG
args = argparse.ArgumentParser('show model')
Args.add_data(args)
Args.add_loss(args)
Args.add_optim(args)
Args.add_model(args)
args = args.parse_args()
##################################################################
# LOAD DATA
yesterdays = [False, True]
rnn_types = ['RNN', 'GRU', 'LSTM']
attn_types = ['dot', 'general', 'mlp']
heads = [1, 2, 4, 8]


def get_loss(model):
    return np.genfromtxt(MODEL_PATH + model + '_loss.txt')


def get_attn(model):

    return np.load(MODEL_PATH + model + '_attn.npy')


def load_loss():
    loss = {}
    for yesterday in yesterdays:
        args.yesterday = yesterday
        for rnn_type in rnn_types:
            args.attn = False
            args.rnn_type = rnn_type
            model = Args.modelname(args)
            loss[model] = get_loss(model)
            args.attn = True
            for attn_type in attn_types:
                args.attn_type = attn_type
                model = Args.modelname(args)
                loss[model] = get_loss(model)
    return loss


def load_attns():
    attns = {}
    for yesterday in yesterdays:
        args.yesterday = yesterday
        # RNN
        args.model = 'RNN'
        args.rnn_type = 'RNN'
        args.attn = True
        args.daytime = False
        for attn_type in ['dot', 'general', 'mlp']:
            args.attn_type = attn_type
            modelname = Args.modelname(args)
            try:
                attns[modelname] = get_attn(modelname)
            except FileNotFoundError:
                print(modelname)
                continue
        # Transformer
        args.model = 'Transformer'
        args.mask_src = True
        args.daytime = True
        for head in heads:
            args.head = head
            modelname = Args.modelname(args)
            try:
                attns[modelname] = get_attn(modelname)
            except FileNotFoundError:
                continue
    return attns



# attn
def plot_attn():
    for model, attn in attns.items():
        path_model = FIG_PATH + model + '/'
        attn.shape


def plot_attns():
    # attentions of context_lengths
    args.attention = True
    for attention_type in attention_types[1:]:
        args.attention_type = attention_type
        path_model = path.join(FIG_PATH, Config.modelname(args))
        for day in range(DAYS_TEST):
            path_day = path.join(path_model, 'Day' + str(day) + '/')
            for length in context_lengths:
                figname = path_day + 'Context' + str(length)
                key = attention_type
                if length != 0:
                    key += str(length)
                Visual.show_attns(attentions_model[key][:, day], args)
                Visual.saveclf(figname)


def compare_attns():
    # compare attention_types
    path_comp = path.join(FIG_PATH, 'Comp')
    for station in range(args.nstation):
        path_station = path.join(path_comp, 'Station' + str(station) + '/')
        flow_max = targets[:, :, station].max()
        for day in range(DAYS_TEST):
            for step in range(args.future):
                figname = path_station + 'Day' + str(day) + 'Step' + str(step)
                Visual.plot_flow_real(targets[:, day, station], flow_max, args)
                for attention_type in attention_types:
                    output = outputs_model[attention_type][:, step, day, station]
                    x = np.arange(output.shape[0]) + step
                    plt.plot(x, output, label=attention_type)
                plt.legend()
                Visual.saveclf(figname)


def dict_name(errs):
    name = ''
    for key in errs.keys():
        name += key
    return name


if __name__ == '__main__':
    attns = load_attns()
    plt.clf()
    reload(Visual)
    # RNN, GRU, LSTM
    errs = {}
    errs['RNNmemory'] = {
        'RNN': [0.1167764,0.13829422,0.17191817,0.21200566,0.25561321,0.29813266,0.33036307,0.35751024],
        'GRU': [0.11842436,0.14016725,0.16910243,0.19837032,0.23616539,0.2783598,0.31488588,0.33411232],
        'LSTM': [0.11940888,0.14337084,0.17014527,0.19695832,0.22289401,0.2486714,0.27041507,0.29585329]}
    # RNN, ATTN
    errs['RNNattn'] = {
        'RNN': [0.1167764,0.13829422,0.17191817,0.21200566,0.25561321,0.29813266,0.33036307,0.35751024],
        'RNNdot': [0.11598904,0.13383965,0.15779936,0.18657154,0.22448166,0.26582742,0.29887876,0.32531753],
        'RNNgeneral': [0.1200269,0.13754445,0.15969662,0.1872087,0.21791759,0.25533512,0.29283288,0.32473403],
        'RNNmlp': [0.11326328,0.13077042,0.15172154,0.17749406,0.20786484,0.23966266,0.26117367,0.27792138]}
    errs['GRUattn'] = {
        'GRU': [0.11842436,0.14016725,0.16910243,0.19837032,0.23616539,0.2783598,0.31488588,0.33411232],
        'GRUdot': [0.11903436,0.14001408,0.16432334,0.19373776,0.23198606,0.27335358,0.31048918,0.33692417],
        'GRUgeneral': [0.12263356,0.14611422,0.1690203,0.19260943,0.21833976,0.24748763,0.2808339,0.30477881],
        'GRUmlp': [0.11564479,0.13653462,0.1619467,0.18842265,0.22107498,0.25619853,0.29012397,0.31232074]}
    errs['LSTMattn'] = {
        'LSTM': [0.11940888,0.14337084,0.17014527,0.19695832,0.22289401,0.2486714,0.27041507,0.29585329],
        'LSTMdot': [0.12420902,0.15058069,0.17603512,0.20085649,0.22806877,0.25199848,0.28018376,0.31460091],
        'LSTMgeneral': [0.12462348,0.15332651,0.18046023,0.21088646,0.23232564,0.25323766,0.27211693,0.29466861],
        'LSTMmlp': [0.11954937,0.14045745,0.16599077,0.19322349,0.21656232,0.23836076,0.25789103,0.28097737]}
    errs['bests'] = {
        'LSTM': [0.11940888,0.14337084,0.17014527,0.19695832,0.22289401,0.2486714,0.27041507,0.29585329],
        'GRUgeneral': [0.12263356,0.14611422,0.1690203,0.19260943,0.21833976,0.24748763,0.2808339,0.30477881],
        'LSTMmlp': [0.11954937,0.14045745,0.16599077,0.19322349,0.21656232,0.23836076,0.25789103,0.28097737],
        'RNNmlp': [0.11326328,0.13077042,0.15172154,0.17749406,0.20786484,0.23966266,0.26117367,0.27792138]}
    errs['Transformer'] = {
        'LSTM': [0.11940888,0.14337084,0.17014527,0.19695832,0.22289401,0.2486714,0.27041507,0.29585329],
        'GRUgeneral': [0.12263356,0.14611422,0.1690203,0.19260943,0.21833976,0.24748763,0.2808339,0.30477881],
        'LSTMmlp': [0.11954937,0.14045745,0.16599077,0.19322349,0.21656232,0.23836076,0.25789103,0.28097737],
        'RNNmlp': [0.11326328,0.13077042,0.15172154,0.17749406,0.20786484,0.23966266,0.26117367,0.27792138],
        'Transformer': [0.1252484,0.14323103,0.16319279,0.17885703,0.19297206,0.21605378,0.24011301,0.26448512]}
    errs['Heads'] = {
        'Transformer1': [0.12742998,0.14555804,0.16556878,0.1805425,0.20246679,0.22880304,0.25942731,0.28205252],
        'Transformer2': [0.12050752,0.13773555,0.15814072,0.17306012,0.19567232,0.22255027,0.25171432,0.27201563],
        'Transformer4': [0.12757941,0.14780284,0.16892248,0.19014184,0.21174681,0.23684862,0.2601923,0.28641814],
        'Transformer8': [0.1252484,0.14323103,0.16319279,0.17885703,0.19297206,0.21605378,0.24011301,0.26448512]}
    errs['Yeseterday'] = {
        'Transformer1': [0.12742998,0.14555804,0.16556878,0.1805425,0.20246679,0.22880304,0.25942731,0.28205252],
        'Transformer2': [0.12050752,0.13773555,0.15814072,0.17306012,0.19567232,0.22255027,0.25171432,0.27201563],
        'Transformer4': [0.12757941,0.14780284,0.16892248,0.19014184,0.21174681,0.23684862,0.2601923,0.28641814],
        'Transformer8': [0.1252484,0.14323103,0.16319279,0.17885703,0.19297206,0.21605378,0.24011301,0.26448512],
        'Transformer1Y': [0.13159131,0.15274382,0.18004076,0.20609488,0.23647431,0.27149594,0.3041555,0.33494282],
        'Transformer2Y': [0.12639542,0.14510071,0.16920826,0.19608609,0.22099708,0.25587004,0.28859895,0.32405019],
        'Transformer4Y': [0.12481638,0.14456406,0.1665625,0.1875757,0.21385154,0.24599187,0.28044811,0.31161097],
        'Transformer8Y': [0.12728627,0.15390354,0.18085256,0.20987818,0.24049622,0.26951972,0.2928308,0.31432039]}

    for v in errs.values():
        Visual.plot_errs(v)
        Visual.saveclf(FIG_PATH + dict_name(v))

    for key, attn in attns.items():
        fig_path = FIG_PATH + key + '/'
        for day in range(3):
            if key.startswith('RNN'):
                Visual.show_attn(attn[:, day], args)
                Visual.saveclf(fig_path + str(day) + '_' + str(head))
            else:
                for head in range(attn.shape[1]):
                    Visual.show_attn(attn[day, head], args)
                    Visual.saveclf(fig_path + str(day) + '_' + str(head))
