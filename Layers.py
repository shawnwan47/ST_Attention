import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Attention import *
from Utils import aeq
from UtilClass import *


class TransformerLayer(nn.Module):
    def __init__(self, dim, head, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.att = MultiHeadAttention(dim, head, dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.feed_forward = PointwiseMLP(dim)

    def forward(self, query, context, mask=None):
        out, att = self.att(query, context, context, mask)
        out = self.layer_norm(query + out)
        out = self.feed_forward(out)
        return out, att


class GaussianMixture(nn.Module):
    def __init__(self, components):
        pass
