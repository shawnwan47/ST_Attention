import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config import Config
from lib import TimeSeriesLoss, MetricDict, mask_target
from lib.io import load_dataset
from builder import build_model


def plot_output(**kwargs):
    config = Config(**kwargs)
    output = np.load(config.result_path.with_suffix('.output'))
    print(output)
    pass


def plot_attn(**kwargs):
    pass


if __name__ == '__main__':
    import fire
    fire.Fire()
