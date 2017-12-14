from random import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import Layers

from Utils import aeq
from UtilClass import BottleLinear


class DayTime(nn.Module):
    def __init__(self, args):
        super(DayTime, self).__init__()
        times = (args.end_time - args.start_time) * 60 // args.gran
        self.daytime = args.daytime
        self.emb_day = nn.Embedding(7, args.day_size)
        self.emb_time = nn.Embedding(times, args.time_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, daytime):
        day = self.dropout(self.emb_day(daytime[:, :, 0]))
        time = self.dropout(self.emb_time(daytime[:, :, 1]))
        return torch.cat((day, time), 2)


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.attn = args.attn
        self.daytime = args.daytime
        self.dim = args.dim
        self.output_size = args.future
        if self.attn:
            self.rnn = Layers.RNNAttn(
                args.rnn_type, args.input_size, args.hidden_size,
                args.num_layers, args.dropout, args.attn_type)
        else:
            self.rnn = Layers.RNNBase(
                args.rnn_type, args.input_size, args.hidden_size,
                args.num_layers, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.linear_out = BottleLinear(args.hidden_size, args.output_size)

    def forward(self, src, input, teach=True):
        src_len, bsz, ndim = src.size()
        tgt_len, bsz_, ndim_ = input.size()
        aeq(bsz, bsz_)
        aeq(ndim, ndim_)
        dim = self.dim
        if self.daytime:
            daytime = input[:, :, dim:]
        hidden = self.rnn.initHidden(src)
        context, hidden = self.rnn.encode(src, hidden)
        out = input[:, :, :dim].clone()
        attns = Variable(torch.zeros(tgt_len, bsz, src_len + tgt_len))
        inp = input[0, :, :dim]
        for i in range(tgt_len):
            if teach and random() < 0.5:
                inp = input[i]
            elif self.daytime:
                inp = torch.cat((inp, daytime[i]), -1)
            inp = inp.unsqueeze(0)
            if self.attn:
                dif, hidden, context, attn = self.rnn(inp, hidden, context)
                attns[i, :, :src_len + i] = attn[0, :, :-1]
            else:
                dif, hidden = self.rnn(inp, hidden)
            out[i] = inp[:, :dim] + self.linear_out(self.dropout(dif))
            inp = out[i].clone()
        if self.attn:
            return out, hidden, attns
        else:
            return out, hidden


class Attn(nn.Module):
    """
    The Attn model for Spatial-Temporal traffic forecasting
    """

    def __init__(self, args):
        super(Attn, self).__init__()
        self.past = args.past
        self.future = args.future
        self.dim = args.dim
        self.head = args.head
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.linear_in = BottleLinear(self.input_size, self.hidden_size)
        self.linear_out = BottleLinear(self.hidden_size, self.output_size)
        self.transformer_layers = nn.ModuleList(
            [Layers.TransformerLayer(self.hidden_size, self.head, args.dropout)
             for _ in range(args.num_layers)])

    def forward(self, inp):
        '''
        inp: batch x len x dim
        out: batch x len - past x dim x future
        attn: batch x len - past x len
        '''
        batch, length, dim = inp.size()
        hid = self.linear_in(inp)
        for layer in self.transformer_layers:
            hid, attn = layer(hid, hid)
        out = self.linear_out(hid[:, self.past:])
        out = out.view(batch, length - self.past, dim, self.future)
        out += inp[self.past:].expand_as(out)
        return out, attn
        # src_len, src_batch, _ = src.size()
        # inp_len, inp_batch, _ = input.size()
        # aeq(src_batch, inp_batch)
        # dim = self.dim
        # if self.daytime:
        #     daytime = input[:, :, dim:]

        # context = self.encode(self.linear_in(src.transpose(0, 1).contiguous()))
        # out = input[:, :, :dim].clone()
        # attn = Variable(torch.zeros(
        #     inp_batch, self.head, inp_len, src_len + inp_len))
        # inp = input[0, :, :dim]

        # for i in range(inp_len):
        #     if teach and random() < 0.5:
        #         inp = input[i]
        #     elif self.daytime:
        #         inp = torch.cat((inp, daytime[i]), -1)
        #     mid = self.linear_in(inp.unsqueeze(0).transpose(0, 1).contiguous())
        #     for layer in self.transformer_layers:
        #         mid, attn[:, :, i, :context.size(1) + 1] = layer(mid, context)
        #     context = torch.cat((context, mid), 1)
        #     out[i] = inp[:, :dim] + self.linear_out(
        #         self.dropout(mid)).transpose(0, 1)
        #     inp = out[i].clone()


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_layers = args.num_layers
        self.gc_in = Layers.GraphConvolution(args.past, args.hidden_size)
        self.gc_hid = Layers.GraphConvolution(args.hidden_size, args.hidden_size)
        self.gc_out = Layers.GraphConvolution(args.hidden_size, args.future)
        self.activation = nn.Sequential(nn.ReLU(), nn.Dropout(args.dropout))

    def forward(self, x, adj):
        x = self.activation(self.gc_in(x, adj))
        for l in range(self.num_layers - 2):
            x = self.activation(self.gc_hid(x, adj))
        return self.gc_out(x, adj)


class GAT(nn.Module):
    '''
    A simplified 2-layer GAT
    '''

    def __init__(self, input_size, hidden_size, head_count, dim):
        super(GAT, self).__init__()
        self.input_size = input_size
        self.dim = dim
        self.hidden_size = hidden_size
        self.head_count = head_count

        self.gat1 = Layers.GraphAttention(input_size, hidden_size, head_count, True)
        self.gat2 = Layers.GraphAttention(self.gat1.dim, dim)

    def forward(self, input, adj):
        out, att1 = self.gat1(input, adj)
        out, att2 = self.gat2(out, adj)
        return out, att1, att2
