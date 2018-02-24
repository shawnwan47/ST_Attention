import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import aeq
from UtilClass import *
import Attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim, head=1, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = Attention.MultiHeadAttention(dim, head, dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.mlp = ResMLP(dim, dropout)

    def forward(self, qry, key, val, mask=None):
        out, att = self.attention(qry, key, val, mask)
        out = self.layer_norm(qry + out)
        out = self.mlp(out)
        return out, att.cpu()


class AttentionLayer(nn.Module):
    def __init__(self, dim, map_type='lin', att_type='dot', res=True, dropout=0.2):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.map_qry = self.mapper(dim, map_type)
        self.map_key = self.mapper(dim, map_type)
        self.map_val = self.mapper(dim, map_type)
        # self.attention = Attention.Attention(dim, att_type, dropout)
        self.res = res
        self.layer_norm = BottleLayerNorm(dim)
        self.resmlp = ResMLP(dim, dropout)

    def mapper(self, dim, map_type):
        if map_type == 'lin':
            return BottleLinear(dim, dim, False)
        elif map_type == 'mlp':
            return MLP(dim, dim, self.dropout)
        elif map_type == 'res':
            return ResMLP(dim, self.dropout)

    def forward(self, qry, key, val, mask=None):
        q, k, v = self.map_qry(qry), self.map_key(key), self.map_val(val)
        out, att = self.attention(q, k, v, mask)
        if self.res:
            out = self.layer_norm(out + qry)
        if self.mlp:
            out = self.resmlp(out)
        return out, att
