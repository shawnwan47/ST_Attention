import torch
import torch.nn as nn
from torch.nn import functional as F

import Layers
from Attention import Attn

import Utils
from UtilClass import *
from Regularizer import *


class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()
        self.past = args.past
        self.future = args.future
        self.dim = args.dim
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.dropout = nn.Dropout(args.dropout)
        self.embed_day = nn.Embedding(7, args.day_size)
        self.embed_time = nn.Embedding(args.daily_times, args.time_size)
        self.adj = args.adj

    def forward(self, inp, daytime=None):
        if daytime is not None:
            daytime = self.embed(daytime)
            return torch.cat((inp, daytime), -1)
        return inp

    def embed(self, daytime):
        if daytime is not None:
            day = self.dropout(self.embed_day(daytime[:, :, 0]))
            time = self.dropout(self.embed_time(daytime[:, :, 1]))
            return torch.cat((day, time), -1)
        return None


class TempLinear(ModelBase):
    def __init__(self, args):
        super(TempLinear, self).__init__(args)
        self.temporal = BottleLinear(self.past, self.future)

    def forward(self, inp, daytime=None):
        out = []
        for i in range(inp.size(1) - self.past):
            inp_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out.append(self.temporal(inp_i).transpose(1, 2))
        out = torch.stack(out, 1)
        return out, self.temporal.weight

    def pack_weight(self, weight):
        return weight.view(1, -1)


class STLinear(TempLinear):
    def __init__(self, args):
        super(STLinear, self).__init__(args)
        self.spatial = BottleSparseLinear(self.dim)

    def forward(self, inp, daytime=None):
        length = inp.size(1)
        out = []
        out_s = self.spatial(inp)
        out = []
        for t in range(length - self.past):
            inp_t = out_s[:, t:t + self.past].transpose(1, 2).contiguous()
            out_t = self.temporal(inp_t).transpose(1, 2)
            out.append(out_t)
        out = torch.stack(out, 1)
        return out, self.temporal.weight


class RNN(ModelBase):
    def __init__(self, args):
        super(RNN, self).__init__(args)
        self.rnn = Layers.RNN(
            args.rnn_type, args.input_size, args.hidden_size,
            args.num_layers, args.dropout)
        self.linear_out = BottleLinear(args.hidden_size, args.output_size)

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
        self.attn = Attn(self.hidden_size, args.attn_type, args.dropout)
        mask = Utils.get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        out = self.forward_rnn(inp, daytime)
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
        self.layers = nn.ModuleList([Layers.HeadAttnLayer(
            self.input_size, self.head, args.dropout)
            for _ in range(self.num_layers)])
        self.linear_out = BottleLinear(self.input_size, self.dim * self.future)
        mask = Utils.get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        out = super(HeadAttn, self).forward(inp, daytime)
        batch, length, dim = out.size()
        attn = []
        for i in range(self.num_layers):
            out, attn_i = self.layers[i](out, self.mask[:length, :length])
            attn.append(attn_i)
        attn = torch.stack(attn, 1)
        out = out[:, self.past:].contiguous()
        out = self.linear_out(out).view(batch, -1, self.future, self.dim)
        return out, attn

    def pack_weight(self, weight):
        return weight.sum(1).view(-1, weight.size(-1))


class TempAttn(ModelBase):
    def __init__(self, args):
        super(TempAttn, self).__init__(args)
        dim, self.pad = Utils.pad_head(self.input_size, args.head)
        hidden_size = dim // args.head
        self.layers = nn.ModuleList([
            Layers.AttnLayer(
                dim, hidden_size, attn_type=args.attn_type,
                head=args.head, merge='cat', dropout=args.dropout
            ) for _ in range(self.num_layers)
        ])
        self.linear = BottleLinear(dim, self.future * self.dim)
        mask = Utils.get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        out = super(TempAttn, self).forward(inp, daytime)
        if self.pad[1]:
            out = F.pad(out, self.pad)
        batch, length, _ = out.size()
        attn = []
        for i in range(self.num_layers):
            out, attn_i = self.layers[i](out, self.mask[:length, :length])
            attn.append(attn_i)
        out = self.linear(out[:, self.past:].contiguous())
        out = out.view(batch, -1, self.future, self.dim)
        attn = torch.stack(attn, 1)
        return out, attn

    def pack_weight(self, weight):
        return weight.sum(1).sum(1).view(-1, weight.size(-1))


class SpatialAttn(ModelBase):
    def __init__(self, args):
        super(SpatialAttn, self).__init__(args)
        self.attn = Layers.AttnLayer(
            self.past, self.hidden_size, attn_type=args.attn_type,
            head=args.head, merge='mean', dropout=args.dropout)
        self.linear = BottleLinear(hidden_size, self.future)
        self.mask = Utils.load_adj() if args.adj else None

    def forward(self, inp, daytime=None):
        batch, length, dim = inp.size()
        out, attn = [], []
        for i in range(length - self.past):
            out_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out_i, attn_i = self.attn(out_i, out_i, self.mask)
            out.append(out_i.transpose(1, 2))
            attn.append(attn_i)
        out = torch.stack(out, 1)
        attn = torch.stack(attn, 1)
        return out, attn


class SpatialAttn2(ModelBase):
    def __init__(self, args):
        super(SpatialAttn2, self).__init__(args)
        self.layer1 = Layers.AttnLayer(
            self.past, args.hidden_size, attn_type=args.attn_type,
            head=args.head, merge='cat', dropout=args.dropout)
        self.layer2 = Layers.AttnLayer(
            args.head * args.hidden_size, self.future, attn_type=args.attn_type,
            head=1, merge='mean', dropout=args.dropout)
        self.adj = args.adj
        self.mask = Utils.load_adj() if args.adj else None

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


class EnTempAttn(ModelBase):
    def __init__(self, args):
        super(EnTempAttn, self).__init__(args)
        assert not args.submodel.startswith('Ens')
        self.subnum = args.subnum
        self.models = nn.ModuleList([
            globals()[args.submodel](args)
            for _ in range(self.subnum)])
        self.attn = BottleLinear(args.day_size + args.time_size,
                                 self.subnum)
        self.softmax = nn.Softmax(2)

    def forward(self, inp, daytime):
        assert daytime is not None
        embedding = super(EnTempAttn, self).embed(daytime)
        batch, length, dim = inp.size()
        out = []
        weight = []
        for i in range(self.subnum):
            out_i = self.models[i](inp, daytime)
            out_i, weight_i = out_i[0], out_i[1]
            out.append(out_i)
            weight.append(weight_i)
        attn = self.softmax(self.attn(embedding[:, self.past:].contiguous()))
        length -= self.past
        out = torch.stack(out, 2).view(batch * length, self.subnum, -1)
        attn = attn.view(batch * length, 1, self.subnum)
        out = torch.bmm(attn, out).view(batch, length, self.future, self.dim)
        weight = torch.stack(weight, 0)
        attn = attn.view(batch, length, self.subnum)
        return out, attn, weight

    def regularizer(self, *params):
        attn, weight = params
        A = []
        for i in range(weight.size(0)):
            A.append(self.models[0].pack_weight(weight[i]))
        A = torch.stack(A, 1)
        orth = orthogonal(A)
        return orth
