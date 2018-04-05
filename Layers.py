import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from Utils import aeq
from UtilClass import *
import Attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim, head, dropout):
        super().__init__()
        self.attention = Attention.MultiHeadAttention(dim, head, dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.mlp = ResMLP(dim, dropout)

    def forward(self, qry, key, val, mask=None):
        out, att = self.attention(qry, key, val, mask)
        out = self.mlp(self.layer_norm(qry + out))
        return out, att


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


class MultiSelfAttnGateLayer(MultiHeadAttentionLayer):
    def __init__(self, dim, head, dropout):
        super().__init__(dim, head, dropout)
        self.gate0 = BottleLinear(dim, dim)
        self.gate1 = BottleLinear(dim, dim)

    def forward(self, data, mask=None):
        '''
        gate: batch x len_q x dim
        '''
        out, att = self.attention(data, data, data, mask)
        gate = F.sigmoid(self.gate0(data) + self.gate1(out))
        out = gate * data + (1 - gate) * out
        out = self.mlp(self.layer_norm(out))
        return out, att, gate.cpu()
