from os import path

import numpy as np
from matplotlib import pyplot as plt

from Config import Config
import Data
import Visual
from Consts import DAYS_TEST, MODEL_PATH, FIG_PATH


# CONFIG
config = Config('PlotResults')
config.add_rnn()
config.add_attention()
config.add_plot()
args = config.parse_args()

##################################################################
# LOAD DATA
targets = np.load(MODEL_PATH + Data.modelname(args) + '_tgt.npy')
# select out flow
station_count = args.ndim // 2
station_order = np.argsort(-targets[:, :, station_count:].sum(0).sum(0))
station_order += station_count
targets = targets[:, :, station_order]

# DATA OF MODELS AND CONTEXTS
att_affix = '_att.npy'
out_affix = '_out.npy'
attention_types = ['RNN', 'dot', 'general', 'mlp']
context_lengths = [0, 8, 16, 32]
outputs_model = {}
attentions_model = {}
# RNN first
args.attention = False
model_path = MODEL_PATH + Data.modelname(args)
outputs_model['RNN'] = np.load(model_path + out_affix)
attentions_model['RNN'] = None
args.attention = True

# attention now
for attention_type in attention_types[1:]:
    args.attention_type = attention_type
    for context_length in context_lengths:
        args.context_length = context_length
        model_path = path.join(MODEL_PATH, Data.modelname(args))

        key = attention_type
        key += str(context_length) if context_length else ''

        attentions = np.load(model_path + att_affix)
        outputs = np.load(model_path + out_affix)[:, :, :, station_order]

        attentions_model[key] = attentions
        outputs_model[key] = outputs

args = config.parse_args()

##################################################################
# PLOT
plt.clf()
# outputs of each attention_types
for attention_type in attention_types:
    if attention_type is 'RNN':
        args.attention = False
    else:
        args.attention = True
        args.attention_type = attention_type
    path_model = path.join(FIG_PATH, Data.modelname(args))

    outputs = outputs_model[attention_type]

    for station in range(args.nstation):
        path_station = path.join(path_model, 'Station' + str(station) + '/')
        flow_max = targets[:, :, station].max()
        for day in range(DAYS_TEST):
            figname = path_station + 'Day' + str(day)
            Visual.plot_flow_real(targets[:, day, station], flow_max, args)
            Visual.plot_flow_steps(outputs[:, :, day, station])
            plt.legend()
            Visual.saveclf(figname)

# attentions of context_lengths
args.attention = True
for attention_type in attention_types[1:]:
    args.attention_type = attention_type
    path_model = path.join(FIG_PATH, Data.modelname(args))
    for day in range(DAYS_TEST):
        path_day = path.join(path_model, 'Day' + str(day) + '/')
        for length in context_lengths:
            figname = path_day + 'Context' + str(length)
            key = attention_type
            if length != 0:
                key += str(length)
            Visual.show_attns(attentions_model[key][:, day], args)
            Visual.saveclf(figname)


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
