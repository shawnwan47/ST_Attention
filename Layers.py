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


class AttnLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 attn_type='mul', head=1, merge='mean', dropout=0.2):
        super(AttnLayer, self).__init__()
        self.head = head
        self.merge = merge
        self.linear = BottleLinear(in_features, out_features, bias=False)
        self.attn = nn.ModuleList([Attn(out_features, attn_type, dropout)
                                   for _ in range(head)])

    def forward(self, inp, mask=None):
        out, attn = [], []
        inp = self.linear(inp)
        for i in range(self.head):
            out_i, attn_i = self.attn[i](inp, mask)
            out.append(out_i)
            attn.append(attn_i)
        attn = torch.stack(attn, 0)
        if self.merge == 'cat':
            out = torch.cat(out, -1)
        elif self.merge == 'mean':
            out = torch.mean(torch.stack(out), 0)
        return out, attn


class HeadAttnLayer(nn.Module):
    def __init__(self, dim, head, dropout=0.2):
        super(HeadAttnLayer, self).__init__()
        self.attn = HeadAttn(dim, head=head, dropout=dropout)
        self.feed_forward = PointwiseMLP(dim, dropout=dropout)

    def forward(self, inp, mask):
        out, attn = self.attn(inp)
        out = self.feed_forward(out)
        return out, attn


class LinearLayer(nn.Module):
    def __init__(self, dim, past, head):
        super(LinearLayer, self).__init__()
        self.past = past
        self.head = head
        self.temporal = nn.ModuleList([
            BottleLinear(past, 1) for _ in range(head)])
        self.spatial = nn.ModuleList([
            BottleSparseLinear(dim, dim) for _ in range(head)])

    def forward(self, inp):
        '''
        inp: batch x length x dim
        out: batch x length - past x head x dim
        '''
        length = inp.size(1)
        out = []
        for i in range(self.head):
            out_i = self.spatial[i](inp)
            out_t = []
            for t in range(length - self.past):
                inp_t = out_i[:, t:t + self.past].transpose(1, 2).contiguous()
                out_t.append(self.temporal[i](inp_t).transpose(1, 2))
            out.append(torch.cat(out_t, -2))
        return torch.stack(out, -2)


class PointwiseMLP(nn.Module):
    ''' A two-layer Feed-Forward-Network.'''

    def __init__(self, dim, dropout=0.1):
        super(PointwiseMLP, self).__init__()
        self.w_1 = BottleLinear(dim, dim)
        self.w_2 = BottleLinear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(dim)

    def forward(self, inp):
        out = self.dropout(self.w_2(self.relu(self.w_1(inp))))
        return self.layer_norm(out + inp)
