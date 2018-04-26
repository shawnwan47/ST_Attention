import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.modules import DayTimeEmbedding


def build_model(args):
    embedding = DayTimeEmbedding(args.time_count, args.time_size, args.day_size)
    if args.framework == 'Seq2Seq':
        encoder = build
