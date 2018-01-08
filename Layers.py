import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Attention import *
from Utils import aeq
from UtilClass import *



class RNNLayer(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        super(RNNLayer, self).__init__()
        assert rnn_type in ['RNN', 'GRU', 'LSTM']
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = getattr(nn, self.rnn_type)(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True)

    def forward(self, inp, hid):
        out, hid = self.rnn(inp, hid)
        return out, hid

    def initHidden(self, inp):
        h = Variable(torch.zeros(
            self.num_layers, inp.size(0), self.hidden_size)).cuda()
        if self.rnn_type == 'LSTM':
            return (h, h.clone())
        else:
            return h


class TransformerLayer(nn.Module):
    def __init__(self, dim_key, dim_val, head, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.att = MultiHeadAttention(dim_key, dim_val, head, dropout)
        self.layer_norm = BottleLayerNorm(dim_key)

    def forward(self, qry, key, val, mask=None):
        out, att = self.att(qry, key, val, mask)
        val_size = val.size(2)
        qry[:, :, :val_size] = qry[:, :, :val_size] + out
        out = self.layer_norm(qry)
        return out, att
