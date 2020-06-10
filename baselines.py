'''
Implementation of baselines:
Historical Average.
Last time continuoue
ARIMA
VAR
SVR
'''
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config import Config
from lib import TimeSeriesLoss, MetricDict, mask_target
from lib.io import load_dataset
