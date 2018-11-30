import math
from random import random

import torch
import torch.nn as nn

from lib.utils import aeq


class Vec2VecBase(nn.Module):
    def __init__(self, embedding, encoder, decoder):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder


class Seq2VecBase(nn.Module):
    def __init__(self, embedding, encoder, decoder):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder


class Seq2SeqBase(nn.Module):
    def __init__(self, embedding, encoder, decoder, history, horizon):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
        self.horizon = horizon
        self.input0 = nn.Parameter(torch.FloatTensor(embedding.size))
        self._reset_input0()

    def _reset_input0(self):
        stdv = 1. / math.sqrt(self.embedding.size)
        self.input0.data.uniform_(-stdv, stdv)

    def _expand_input0(self, input):
        return self.input0.expand_as(input[:, [0]])

    def _check_args(self, data, time, day):
        data_length = self.history + self.horizon - 1
        aeq(data_length, data.size(1), time.size(1), day.size(1))
