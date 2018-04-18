import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from Utils import aeq
from UtilClass import *
import Attention


class TransformerLayer(nn.Module):
    def __init__(self, head, dim, dropout):
        super().__init__()
        self.attention = Attention.MultiHeadedAttention(head, dim, dropout)
        self.layer_norm = LayerNorm(dim)
        self.mlp = ResMLP(dim, dropout)

    def forward(self, data, mask=None):
        out, att = self.attention(data, data, data, mask)
        out = self.mlp(self.layer_norm(data + out))
        return out, att


class TransformerGateLayer(TransformerLayer):
    def __init__(self, head, dim, dropout):
        super().__init__(head, dim, dropout)
        self.gate_data = nn.Linear(dim, dim)
        self.gate_context = nn.Linear(dim, dim)

    def forward(self, data, mask=None):
        '''
        gate: batch x len_q x dim
        '''
        out, att = self.attention(data, data, data, mask)
        gate = F.sigmoid(self.gate_data(data) + self.gate_context(out))
        out = gate * data + (1 - gate) * out
        out = self.mlp(self.layer_norm(out))
        return out, att, gate.cpu()

class TransformerFusionLayer(TransformerGateLayer):
    def __init__(self, head, dim, dropout):
        super().__init__(head, dim, dropout)
        self.gate_data = nn.Linear(dim, 1)
        self.gate_context = nn.Linear(dim, 1)

    def forward(self, data, mask=None):
        out, att = self.attention(data, data, data, mask)
        gate = F.sigmoid(self.gate_data(data) + self.gate_context(out))
        out = gate * data + (1 - gate) * out
        out = self.mlp(self.layer_norm(out))
        return out, att, gate.cpu()
