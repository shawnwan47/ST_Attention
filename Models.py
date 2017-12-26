import torch
import torch.nn as nn

import Layers

from Utils import get_mask_trim, get_mask_dilated, aeq
from UtilClass import *


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
        self.embedding_day = nn.Embedding(7, args.day_size)
        self.embedding_time = nn.Embedding(args.daily_times, args.time_size)
        self.adj = args.adj
        # self.longlat = args.longlat
        self.eval_layers = self.num_layers

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

    def fix_layers(self):
        self.embedding_day.require_grad = False
        self.embedding_time.require_grad = False

    def set_layers(self, eval_layers=None):
        if eval_layers is not None:
            self.eval_layers = eval_layers

    def reset(self):
        self.eval_layers = self.num_layers
        self.embedding_day.require_grad = True
        self.embedding_time.require_grad = True


class RNN(ModelBase):
    def __init__(self, args):
        super(RNN, self).__init__(args)
        self.layer = Layers.RNNBase(
            args.rnn_type, args.input_size, args.hidden_size,
            args.num_layers, args.dropout)
        self.linear_out = BottleLinear(args.hidden_size, args.output_size)

    def forward(self, inp):
        '''
        inp: batch x len x input_size
        out: batch x len - past x future x dim
        attn: batch x len - past x len
        '''
        residual = inp[:, self.past:, :self.dim].unsqueeze(-2)
        hid = self.layer.initHidden(inp)
        out, hid = self.layer(inp, hid)
        out = self.linear_out(self.dropout(out[:, self.past:]))
        batch, length, dim = out.size()
        out = out.view(batch, length, self.future, self.dim)
        out += residual
        return out


class RNNAttn(ModelBase):
    def __init__(self, args):
        self.layer = Layers.RNNAttn(
            args.rnn_type, args.input_size, args.hidden_size,
            args.num_layers, args.dropout, args.attn_type)
        self.linear_out = BottleLinear(args.hidden_size, args.output_size)
        mask = get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        inp, daytime = super(RNNAttn, self).forward(inp, daytime)
        hid = self.layer.initHidden(inp)
        mask = self.mask[:inp.size(1), :inp.size(1)]
        out, hid, attn = self.layer(inp, hid, mask)
        out = self.linear_out(self.dropout(out[:, self.past:]))
        batch, length, dim = out.size()
        out = out.view(batch, length, self.future, self.dim)
        out += residual
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
        out = out[:, self.past:]
        out = self.linear_out(out.contiguous())
        out = out.view(batch, -1, self.future, dim)
        out = out[:, :, :, :self.dim]
        out += inp[:, self.past:].unsqueeze(-2)
        return out, attn


class ConvAttn(ModelBase):
    def __init__(self, args):
        super(ConvAttn, self).__init__(args)
        channel = args.channel
        self.layer_in = Layers.ConvAttnLayer(
            self.input_size, 1, channel, args.attn_type, args.value_proj)
        self.layer_out = Layers.ConvAttnLayer(
            self.input_size, channel, self.future, args.attn_type, args.value_proj)
        self.register_buffer('mask', get_mask_trim(args.input_length, self.past))

    def forward(self, inp, daytime=None):
        '''
        inp: batch x length x dim
        out: batch x length - past x future x dim
        attn: num_layers x out_channel x in_channel x batch x length x length
        '''
        inp = super(ConvAttn, self).forward(inp, daytime)
        batch, length, dim = inp.size()
        out = inp.unsqueeze(2)
        mask = self.mask[:length, :length]
        out, attn_in = self.layer_in(out, mask)
        out, attn_out = self.layer_out(out, mask)
        # out = out + inp.unsqueeze(2)
        out = out[:, self.past:, :, :self.dim]
        return out, attn_in, attn_out


class SpatialAttn(ModelBase):
    def __init__(self, args):
        super(SpatialAttn, self).__init__(args)
        self.adj = args.adj
        self.channel = args.channel

    def forward(self, inp):
        batch, length, dim = inp.size()
        out = inp.transpose(1, 2).contiguous()
        return out


class Linear(ModelBase):
    def __init__(self, args):
        super(Linear, self).__init__(args)
        self.channel = args.channel
        self.layer = Layers.LinearLayer(
            self.input_size, self.past, self.channel)
        self.linear_out = nn.Linear(self.channel, self.future)

    def forward(self, inp, daytime=None):
        inp = super(Linear, self).forward(inp, daytime)
        out = self.layer(inp)
        out = self.linear_out(out.transpose(2, 3).contiguous()).transpose(2, 3)
        return out


class LinearAttn(ModelBase):
    def __init__(self, args):
        super(LinearAttn, self).__init__(args)
        self.channel = args.channel
        self.layer = Layers.LinearAttnLayer(
            self.input_size, self.past, self.channel, self.future, dropout=args.dropout)

    def forward(self, inp, daytime=None):
        inp = super(LinearAttn, self).forward(inp, daytime)
        return self.layer(inp)


class LinearTemporal(ModelBase):
    def __init__(self, args):
        super(LinearTemporal, self).__init__(args)
        self.linear = BottleLinear(self.past, self.future)

    def forward(self, inp, daytime=None):
        out = []
        for i in range(inp.size(1) - self.past):
            inp_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out.append(self.linear(inp_i).transpose(1, 2))
        out = torch.stack(out, 1)
        return out


class LinearSpatial(ModelBase):
    def __init__(self, args):
        super(LinearSpatial, self).__init__(args)
        self.linear = BottleLinear(self.input_size, self.output_size)

    def forward(self, inp):
        '''
        inp: batch x length x dim
        out: batch x length - past x dim
        '''
        batch, _, dim = inp.size()
        out = self.linear(inp)[:, self.past:]
        out = out.contiguous().view(batch, -1, self.future, dim)
        return out


class LinearSpatialTemporal(ModelBase):
    def __init__(self, args):
        super(LinearSpatialTemporal, self).__init__(args)
        self.temporal = BottleLinear(self.past, self.future)
        self.spatial = BottleLinear(self.input_size, self.input_size)

    def forward(self, inp):
        batch, length, dim = inp.size()
        out = []
        for i in range(length - self.past):
            inp_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out_i = self.temporal(inp_i).transpose(1, 2).contiguous()
            out_i = self.spatial(out_i)
            out.append(out_i)  # batch x future x dim
        out = torch.stack(out, 1)
        out = out.view(batch, -1, self.future, dim)
        return out
