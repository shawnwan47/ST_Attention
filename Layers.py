import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Attention import *
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
    def __init__(self, dim, head, dropout=0.2):
        super(HeadAttnLayer, self).__init__()
        self.attn = MultiHeadedAttention(dim, head=head, dropout=dropout)
        self.feed_forward = PointwiseMLP(dim, dropout=dropout)

    def forward(self, inp, mask):
        out, attn = self.attn(inp)
        # out = self.feed_forward(out)
        return out, attn


class ConvAttnLayer(nn.Module):
    def __init__(self, dim, head_in, head_out, attn_type, dropout=0.2):
        super(ConvAttnLayer, self).__init__()
        self.head_in = head_in
        self.head_out = head_out
        self.attn = nn.ModuleList([
            nn.ModuleList([AttnInterface(
                dim, attn_type, dropout=dropout)
                for _ in range(head_in)])
            for _ in range(head_out)])

    def forward(self, inp, mask=None):
        '''
        IN
        inp: batch x length x head_in x dim
        mask: length x length
        OUT
        out: batch x length x head_out x dim
        attn: head_out x head_in x batch x length x length
        '''
        batch, length, head_in, dim = inp.size()

        out = []
        attn = []
        for i in range(self.head_out):
            out_i = []
            attn_i = []
            for j in range(self.head_in):
                out_j, attn_j = self.attn[i][j](inp[:, :, j], mask)
                out_i.append(out_j)
                attn_i.append(attn_j)
            attn_i = torch.stack(attn_i, 0)
            attn.append(attn_i)
            out_i = torch.mean(torch.stack(out_i), 0)
            out.append(out_i)
        out = torch.stack(out, -2)
        attn = torch.stack(attn, 0)
        return out, attn


class WAttnLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 attn_type='mul', head=1, merge='cat', dropout=0.2):
        super(WAttnLayer, self).__init__()
        self.head = head
        self.attn = nn.ModuleList([
            WAttnInterface(in_features, out_features, attn_type, dropout)
            for _ in range(head)])
        self.merge = merge

    def forward(self, inp, mask=None):
        out, attn, weight = [], [], []
        for i in range(self.head):
            out_i, attn_i, weight_i = self.attn[i](inp)
            out.append(out_i)
            attn.append(attn_i)
            weight.append(weight_i)
        attn = torch.stack(attn, 0)
        weight = torch.stack(weight, 0)
        if self.merge == 'cat':
            out = torch.cat(out, -1)
        elif self.merge == 'mean':
            out = torch.stack(out)
            out = torch.mean(out, 0)
        return out, attn, weight


class SelfAttnLayer(nn.Module):
    def __init__(self, dim, head_in, head_out, dropout):
        super(SelfAttnLayer, self).__init__()
        self.head_in = head_in
        self.head_out = head_out
        self.dim = dim
        self.layer = SelfAttention(dim, head_out, dropout)

    def forward(self, inp):
        assert inp.dim() < 5
        out = inp.view(-1, self.head_in, self.dim)
        out, attn = self.layer(out)
        out = out.view(inp.size(0), -1, self.head_out, self.dim)
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


class LinearAttnLayer(LinearLayer):
    def __init__(self, dim, past, head, hop, dropout=0.2):
        super(LinearAttnLayer, self).__init__(dim, past, head)
        self.attn = SelfAttention(dim, hop=hop, dropout=dropout)

    def forward(self, inp):
        out = super(LinearAttnLayer, self).forward(inp)
        batch, length, head, dim = out.size()
        out, attn = self.attn(out.view(-1, head, dim))
        out = out.view(batch, length, -1, dim)
        attn = attn.view(batch, length, -1, head)
        return out, attn


class PointwiseMLP(nn.Module):
    ''' A two-layer Feed-Forward-Network.'''

    def __init__(self, dim, adj=None, dropout=0.1):
        super(PointwiseMLP, self).__init__()
        self.w_1 = BottleLinear(dim, dim)
        self.w_2 = BottleLinear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(dim)

    def forward(self, inp):
        out = self.dropout(self.w_2(self.relu(self.w_1(inp))))
        return self.layer_norm(out + inp)
