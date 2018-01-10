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
        self.att = MultiHeadAttention(dim, dim, head, dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.feed_forward = PointwiseMLP(dim)

    def forward(self, query, context, mask=None):
        out, att = self.att(query, context, context, mask)
        out = self.layer_norm(query + out)
        out = self.feed_forward(out)
        return out, att


class Transformer2Layer(nn.Module):
    def __init__(self, dim_key, dim_val, head, dropout=0.1):
        super(Transformer2Layer, self).__init__()
        self.att = MultiHeadAttention(dim_key, dim_val, head, dropout)

    def forward(self, qry, key, val, mask=None):
        out_val, att = self.att(qry, key, val, mask)
        out = Variable(qry.data, volatile=qry.volatile)
        out[:, :, :val.size(-1)] = out_val
        return out, att
