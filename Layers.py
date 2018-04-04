import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from Utils import aeq
from UtilClass import *
import Attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim, head, dropout=0.2):
        super().__init__()
        self.attention = Attention.MultiHeadAttention(dim, head, dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.mlp = ResMLP(dim, dropout)

    def forward(self, qry, key, val, mask=None):
        out, att = self.attention(qry, key, val, mask)
        out = self.layer_norm(qry + out)
        out = self.mlp(out)
        return out, att.cpu()


class AttentionFusionLayer(nn.Module):
    def __init__(self, dim, head, att_type, dropout=0.2):
        super().__init__()
        self.attention = Attention.AttentionFusion(dim, head, att_type, dropout)
        self.gate_query = BottleLinear(dim, 1)
        self.gate_context = BottleLinear(dim, 1)
        self.layer_norm = LayerNorm(dim)
        self.mlp = ResMLP(dim, dropout)

    def forward(self, query, context, mask):
        '''
        out, query, context: batch x num x features
        gate: batch x num x 1
        att_fusion: batch x num x head
        att_head: batch x num x head x num_key
        '''
        context, att_head, att_fusion = self.attention(query, context, context, mask)
        gate = F.sigmoid(self.gate_query(query) + self.gate_context(context))
        out = gate * query + (1 - gate) * context
        out = self.layer_norm(out)
        out = self.mlp(out)
        return out, gate.cpu(), att_fusion, att_head


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
