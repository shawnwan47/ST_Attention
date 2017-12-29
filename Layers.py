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
    def __init__(self, in_features, out_features, head=1,
                 attn_type='dot', merge_type='add',
                 merge='cat', dropout=0.1):
        super(AttnLayer, self).__init__()
        self.head = head
        self.merge = merge
        self.linear = BottleLinear(in_features, out_features, bias=False)
        # self.attn = nn.ModuleList([
        #     MergeAttn(out_features, attn_type, merge_type, dropout)
        #     for _ in range(head)])
        self.attn = nn.ModuleList([
            Attn(out_features, attn_type, dropout)
            for _ in range(head)])

    def forward(self, inp, mask=None):
        out, attn = [], []
        inp = self.linear(inp)
        for i in range(self.head):
            out_i, attn_i = self.attn[i](inp, inp, mask)
            out.append(out_i)
            attn.append(attn_i)
        attn = torch.stack(attn, 1)
        if self.merge == 'cat':
            out = torch.cat(out, -1)
        elif self.merge == 'mean':
            out = torch.mean(torch.stack(out), 0)
        return out, attn


class HeadAttnLayer(nn.Module):
    def __init__(self, dim, head, dropout=0.1):
        super(HeadAttnLayer, self).__init__()
        self.attn = HeadAttn(dim, head=head, dropout=dropout)
        self.feedforward = PointwiseMLP(dim, dropout)

    def forward(self, inp, mask):
        out, attn = self.attn(inp)
        out = self.feedforward(out)
        return out, attn


class GenMat(nn.Module):
    pass
