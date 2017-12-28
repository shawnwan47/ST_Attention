import torch
import torch.nn as nn

import Layers
from Attention import Attn, LinearAttn

from Utils import get_mask_trim, get_mask_dilated, aeq, load_adj
from UtilClass import *


class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()
        self.past = args.past
        self.future = args.future
        self.dim = args.dim
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.dropout = nn.Dropout(args.dropout)
        self.embedding_day = nn.Embedding(7, args.day_size)
        self.embedding_time = nn.Embedding(args.daily_times, args.time_size)
        self.adj = args.adj

    def forward(self, inp, daytime=None):
        if daytime is not None:
            batch, length, dim = inp.size()
            batch_, length_, dim_ = daytime.size()
            aeq(batch, batch_)
            aeq(length, length_)
            aeq(dim_, 2)
            day = self.dropout(self.embedding_day(daytime[:, :, 0]))
            time = self.dropout(self.embedding_time(daytime[:, :, 1]))
            inp = torch.cat((inp, day, time), -1)
        return inp.contiguous()


class TemporalLinear(ModelBase):
    def __init__(self, args):
        super(TemporalLinear, self).__init__(args)
        self.linear = BottleLinear(self.past, self.future)

    def forward(self, inp, daytime=None):
        out = []
        for i in range(inp.size(1) - self.past):
            inp_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out.append(self.linear(inp_i).transpose(1, 2))
        out = torch.stack(out, 1)
        return out


class RNN(ModelBase):
    def __init__(self, args):
        super(RNN, self).__init__(args)
        self.rnn = Layers.RNNLayer(
            args.rnn_type, args.input_size, args.input_size,
            args.num_layers, args.dropout)
        self.linear_out = BottleLinear(args.input_size, args.output_size)

    def forward_rnn(self, inp, daytime=None):
        inp = super(RNN, self).forward(inp, daytime)
        hid = self.rnn.initHidden(inp)
        out, _ = self.rnn(inp, hid)
        return out

    def forward(self, inp, daytime=None):
        out = self.forward_rnn(inp, daytime)
        batch, length, _ = out.size()
        out = self.linear_out(self.dropout(out[:, self.past:]))
        out = out.view(batch, -1, self.future, self.dim)
        return out


class RNNAttn(RNN):
    def __init__(self, args):
        super(RNNAttn, self).__init__(args)
        self.attn = Attn(self.input_size, args.attn_type, args.dropout)
        mask = get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        out = self.forward_rnn(inp, daytime).contiguous()
        batch, length, dim = out.size()
        mask = self.mask[:length, :length]
        out, attn = self.attn(out, mask)
        out = self.linear_out(self.dropout(out[:, self.past:]))
        out = out.view(batch, -1, self.future, self.dim)
        return out, attn


class HeadAttn(ModelBase):
    def __init__(self, args):
        super(HeadAttn, self).__init__(args)
        self.head = args.head
        self.dilated = args.dilated
        self.dilation = args.dilation[:self.num_layers + 1]
        self.layers = nn.ModuleList([Layers.HeadAttnLayer(
            self.input_size, self.head, args.dropout)
            for _ in range(self.num_layers)])
        self.linear_out = BottleLinear(self.dim, self.dim * self.future)
        if self.dilated:
            mask = get_mask_dilated(args.input_length, self.dilation)
        else:
            mask = get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        out = super(HeadAttn, self).forward(inp, daytime)
        batch, length, dim = out.size()
        attn = []
        for i in range(self.num_layers):
            mask = self.mask[i] if self.dilated else self.mask
            mask = mask[:length, :length]
            out, attn_i = self.layers[i](out, mask)
            attn.append(attn_i)
        attn = torch.cat(attn, 1)
        out = out[:, self.past:].contiguous()
        out = self.linear_out(out).view(batch, -1, self.future, dim)
        out = out[:, :, :, :self.dim] + inp[:, self.past:].unsqueeze(-2)
        return out, attn


class TemporalAttn(ModelBase):
    def __init__(self, args):
        super(TemporalAttn, self).__init__(args)
        hidden_size = self.input_size // args.head
        self.attn = Layers.AttnLayer(
            self.input_size, hidden_size, attn_type=args.attn_type,
            head=args.head, merge='cat', dropout=args.dropout
        )
        self.dropout = nn.Dropout(args.dropout)
        self.linear = BottleLinear(args.head * hidden_size,
                                   self.future * self.dim)
        mask = get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        inp = super(TemporalAttn, self).forward(inp, daytime)
        batch, length, _ = inp.size()
        out, attn = self.attn(inp, self.mask[:length, :length])
        out = self.dropout(out[:, self.past:]).contiguous()
        out = self.linear(out).view(batch, -1, self.future, self.dim)
        return out, attn


class SpatialAttn(ModelBase):
    def __init__(self, args):
        super(SpatialAttn, self).__init__(args)
        hidden_size = self.past // self.head
        self.attn = Layers.AttnLayer(
            self.past, hidden_size, attn_type=args.attn_type,
            head=args.head, merge='mean', dropout=args.dropout)
        self.linear = BottleLinear(hidden_size, self.future)
        self.mask = load_adj() if args.adj else None

    def forward(self, inp, daytime=None):
        batch, length, dim = inp.size()
        out, attn = [], []
        for i in range(length - self.past):
            out_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out_i, attn_i = self.attn(out_i, self.mask)
            out.append(out_i.transpose(1, 2))
            attn.append(attn_i)
        out = torch.stack(out, 1)
        attn = torch.stack(attn, 1)
        return out, attn


class SpatialAttn2(ModelBase):
    def __init__(self, args):
        super(SpatialAttn2, self).__init__(args)
        hidden_size = self.past // args.head
        self.layer1 = Layers.AttnLayer(
            self.past, hidden_size, attn_type=args.attn_type,
            head=args.head, merge='cat', dropout=args.dropout)
        self.layer2 = Layers.AttnLayer(
            args.head * hidden_size, self.future, attn_type=args.attn_type,
            head=1, merge='mean', dropout=args.dropout)
        self.adj = args.adj
        self.mask = load_adj() if args.adj else None

    def forward(self, inp, daytime=None):
        batch, length, dim = inp.size()
        out, attn1, attn2 = [], [], []
        for i in range(length - self.past):
            out_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out_i, attn1_i, weight1 = self.layer1(out_i, self.mask)
            out_i, attn2_i, weight2 = self.layer2(out_i, self.mask)
            out.append(out_i.transpose(1, 2))
            attn1.append(attn1_i)
            attn2.append(attn2_i)
        out = torch.stack(out, 1)
        attn1 = torch.stack(attn1, 1)
        attn2 = torch.stack(attn2, 1)
        return out, attn1, attn2, weight1, weight2


class Linear(ModelBase):
    def __init__(self, args):
        super(Linear, self).__init__(args)
        self.head = args.head
        self.layer = Layers.LinearLayer(
            self.dim, self.past, self.head)
        self.linear_out = nn.Linear(self.head, self.future)

    def forward(self, inp, daytime=None):
        out = self.layer(inp)
        out = self.linear_out(out.transpose(2, 3).contiguous()).transpose(2, 3)
        return out
