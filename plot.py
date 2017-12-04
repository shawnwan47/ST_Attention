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
args.attention = True

# LOAD DATA
modelname = Data.modelname(args)
modelpath = MODEL_PATH + modelname
figpath = FIG_PATH + modelname

targets = np.load(modelpath + '_tgt.npy')
outputs = np.load(modelpath + '_out.npy')
attentions = np.load(modelpath + '_att.npy')

# show predictions
station_flow = targets.sum(0).sum(0)
idx = np.argsort(-station_flow)
idx_sel = np.arange(args.ndim)[idx == 7][0]
for day in range(DAYS_TEST):
    Visual.plot_predictions(targets[:, day, idx_sel],
                            outputs[:, :, day, idx_sel], args)
    plt.savefig(figpath + str(day) + '.png')

# show attentions
Visual.show_attns(attentions[:, 1], args)
plt.savefig(figpath + '_att.png')
