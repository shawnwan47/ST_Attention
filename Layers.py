import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Attention import AttentionInterface, SelfAttention
from Utils import aeq
from UtilClass import *



class RNNBase(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        super(RNNBase, self).__init__()
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


class RNNAttn(RNNBase):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout,
                 attn_type):
        super(RNNAttn, self).__init__(
            rnn_type, input_size, hidden_size, num_layers, dropout)
        self.attn_type = attn_type
        self.attn = GlobalAttention(hidden_size, attn_type)

    def forward(self, inp, hid, mask):
        out, hid = self.rnn(inp, hid)
        out, attn = self.attn(out, mask)
        return out, hid, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head, channel, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.channel = channel
        self.attn = nn.ModuleList([
            AttentionInterface(dim, 'head', head=head, dropout=dropout)
            for _ in range(channel)])
        self.feed_forward = PointwiseMLP(dim, dropout=dropout)

    def forward(self, inp, mask):
        batch, length, channel, dim = inp.size()
        out = []
        attn = []
        for i in range(self.channel):
            out_i, attn_i = self.attn[i](inp[:, :, i], mask)
            out_i = self.feed_forward(out_i)
            out.append(out_i)
            attn.append(attn_i)
        out = torch.stack(out, -2)
        attn = torch.stack(attn, 1)
        return out, attn


class MultiChannelAttention(nn.Module):
    def __init__(self, dim, in_channel, out_channel, attn_type, dropout=0.1):
        super(MultiChannelAttention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.attn_merge = nn.ModuleList([
            AttentionInterface(dim, 'self', dropout=dropout)
            for _ in range(out_channel)])
        self.attn = nn.ModuleList([
            nn.ModuleList([
                AttentionInterface(dim, attn_type, dropout=dropout)
                for _ in range(in_channel)])
            for _ in range(out_channel)])

    def forward(self, inp, mask=None):
        '''
        IN
        inp: batch x length x in_channel x dim
        mask: length x length
        OUT
        out: batch x length x out_channel x dim
        attn_merge: batch x length x out_channel x in_channel
        attn: batch x length x out_channel x in_channel x length
        '''
        batch, length, in_channel, dim = inp.size()

        out = []
        attn = []
        attn_merge = []
        for i in range(self.out_channel):
            out_i = []
            attn_i = []
            for j in range(self.in_channel):
                out_j, attn_j = self.attn[i][j](inp[:, :, j], mask)
                out_i.append(out_j)
                attn_i.append(attn_j)
            attn_i = torch.stack(attn_i, 2)
            out_i = torch.stack(out_i, 2)
            out_i = out_i.view(-1, in_channel, dim)
            out_i, attn_merge_i = self.attn_merge[i](out_i)
            out.append(out_i.view(batch, length, dim))
            attn.append(attn_i)
            attn_merge.append(attn_merge_i)
        out = torch.stack(out, 2)
        attn = torch.stack(attn, 2)
        attn_merge = torch.stack(attn_merge, 2)
        return out, attn, attn_merge

