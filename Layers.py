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


class HeadAttnLayer(nn.Module):
    def __init__(self, dim, head, channel, dropout=0.2):
        super(HeadAttnLayer, self).__init__()
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


class ConvAttnLayer(nn.Module):
    def __init__(self, dim, in_channel, out_channel, attn_type, value_proj=False, dropout=0.2):
        super(ConvAttnLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.attn_conv = nn.ModuleList([
            nn.ModuleList([
                AttentionInterface(dim, attn_type, value_proj=value_proj, dropout=dropout)
                for _ in range(in_channel)])
            for _ in range(out_channel)])
        self.linear_pool = nn.ModuleList([
            BottleLinear(in_channel, 1) for _ in range(out_channel)])
        self.attn_pool = nn.ModuleList([
            AttentionInterface(dim, 'self', value_proj=value_proj, dropout=dropout)])

    def forward(self, inp, mask=None):
        '''
        IN
        inp: batch x length x in_channel x dim
        mask: length x length
        OUT
        out: batch x length x out_channel x dim
        attn_conv: out_channel x in_channel x batch x length x length
        '''
        batch, length, in_channel, dim = inp.size()

        out = []
        attn_conv = []
        for i in range(self.out_channel):
            out_i = []
            attn_i = []
            for j in range(self.in_channel):
                out_j, attn_j = self.attn_conv[i][j](inp[:, :, j], mask)
                out_i.append(out_j)
                attn_i.append(attn_j)
            attn_i = torch.stack(attn_i, 0)
            out_i = torch.stack(out_i, -1)
            # out_i = torch.sum(out_i, -1)
            out_i = self.linear_pool[i](out_i).squeeze(-1)
            out.append(out_i)
            attn_conv.append(attn_i)
        out = torch.stack(out, -2)
        attn_conv = torch.stack(attn_conv, 0)
        return out, attn_conv


class LinearLayer(nn.Module):
    def __init__(self, dim, past, channel, dropout=0.2):
        super(LinearLayer, self).__init__()
        self.past = past
        self.channel = channel
        self.temporal = nn.ModuleList([
            BottleLinear(past, 1) for _ in range(channel)])
        self.spatial = nn.ModuleList([
            BottleLinear(dim, dim) for _ in range(channel)])

    def forward(self, inp):
        length = inp.size(1)
        out = []
        for i in range(self.channel):
            out_i = self.spatial[i](inp)
            out_t = []
            for t in range(length - self.past):
                inp_t = out_i[:, t:t + past].transpose(1, 2).contiguous()
                out_t.append(self.temporal[i](inp_t).transpose(1, 2))
            out.append(torch.cat(out_t, -2))
        return torch.stack(out, -2)
