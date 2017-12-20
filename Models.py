import torch
import torch.nn as nn

import Layers

from Utils import get_mask_trim, get_mask_dilated
from UtilClass import *


class Embedding_DayTime(nn.Module):
    def __init__(self, args):
        super(Embedding_DayTime, self).__init__()
        self.embedding_day = nn.Embedding(7, args.day_size)
        self.embedding_time = nn.Embedding(args.daily_times, args.time_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, daytime):
        day = self.dropout(self.embedding_day(daytime[:, :, 0]))
        time = self.dropout(self.embedding_time(daytime[:, :, 1]))
        return torch.cat((day, time), -1)

    def fix(self):
        self.embedding_day.require_grad = False
        self.embedding_time.require_grad = False

    def reset(self):
        self.embedding_day.require_grad = True
        self.embedding_time.require_grad = True


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.past = args.past
        self.future = args.future
        self.attn = args.attn
        self.dim = args.dim
        self.input_size = args.input_size
        self.output_size = args.output_size
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
        mask = get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp):
        '''
        inp: batch x len x input_size
        out: batch x len - past x future x dim
        attn: batch x len - past x len
        '''
        residual = inp[:, self.past:, :self.dim].unsqueeze(-2)
        hid = self.rnn.initHidden(inp)
        if self.attn:
            mask = self.mask[:inp.size(1), :inp.size(1)]
            out, hid, attn = self.rnn(inp, hid, mask)
        else:
            out, hid = self.rnn(inp, hid)
        out = self.linear_out(self.dropout(out[:, self.past:]))
        batch, length, dim = out.size()
        out = out.view(batch, length, self.future, self.dim)
        out += residual
        if self.attn:
            return out, attn
        else:
            return out


class Attn(nn.Module):
    def __init__(self, args, adj=None):
        super(Attn, self).__init__()
        self.past = args.past
        self.future = args.future
        self.dim = args.dim
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.linear_out = BottleLinear(self.input_size, self.output_size)
        self.layers = nn.ModuleList([
            Layers.AttnLayer(self.input_size, args.dropout)
            for _ in range(self.num_layers)])
        self.dilated = args.dilated
        self.dilation = args.dilation[:self.num_layers + 1]
        if self.dilated:
            masks = []
            for i in range(self.num_layers):
                dilation = self.dilation[i]
                window = self.dilation[i + 1] // dilation
                mask = _get_mask_dilated(args.input_length, dilation, window)
                masks.append(mask)
            self.register_buffer('mask', torch.stack(masks, 0))
        else:
            mask = get_mask_trim(args.input_length, self.past)
            self.register_buffer('mask', mask)

    def forward(self, inp):
        '''
        inp: batch x len x input_size
        out: batch x len - past x future x dim
        attn: batch x len - past x len
        '''
        residual = inp[:, self.past:, :self.dim].unsqueeze(-2)
        out = inp.clone()
        attns = []
        for i in range(self.num_layers):
            if self.dilated:
                mask = self.mask[i, :inp.size(1), :inp.size(1)]
            else:
                mask = self.mask[:inp.size(1), :inp.size(1)]
            out, attn = self.layers[i](out, mask)
            attns.append(attn)
        out = self.linear_out(out[:, self.past:].contiguous())
        batch, length, _ = out.size()
        out = out.view(batch, length, self.future, self.dim)
        out += residual
        attn = torch.stack(attns, 1)
        return out, attn


class STAttn(nn.Module):
    def __init__(self, args, adj=None):
        super(STAttn, self).__init__()
        self.past = args.past
        self.future = args.future
        self.dim = args.dim
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.channel = args.channel
        self.dilated = args.dilated
        self.dilation = args.dilation[:self.num_layers + 1]
        self.dropout = nn.Dropout(args.dropout)
        self.linear_out = BottleSparseLinear(
            self.input_size, self.output_size, adj=adj)
        self.encoders = nn.ModuleList([
            Layers.MultiChannelSelfAttention(
                self.input_size, self.channel, adj, args.dropout)
            for _ in range(self.num_layers)])
        self.decoders = nn.ModuleList([
            Layers.MultiChannelAttention(
                self.input_size, self.channel, adj, args.dropout)
            for _ in range(self.num_layers)])
        if self.dilated:
            mask = get_mask_dilated(args.input_length, self.dilation)
        else:
            mask = get_mask_trim(args.input_length, self.past)
        self.register_buffer('mask', mask)
        self.eval_layers = self.num_layers

    def fix_layers(self, layers):
        for i in range(layers):
            self.encoders[i].require_grad = False
            self.decoders[i].require_grad = False

    def set_eval_layers(self, layers):
        self.eval_layers = layers

    def reset(self):
        self.eval_layers = self.num_layers
        for i in range(self.num_layers):
            self.encoders[i].require_grad = True
            self.decoders[i].require_grad = True

    def forward(self, inp):
        '''
        inp: batch x length x dim
        out: batch x length - past x future x dim
        attn_merge: batch x num_layers x length x channel
        attn_channel: batch x num_layers x channel x length - past x length
        attn_context: batch x num_layers x channel x length x length
        '''
        res = inp[:, self.past:, :self.dim].unsqueeze(-2)
        out = inp[:, self.past:]
        context = inp.unsqueeze(1).repeat(1, self.channel, 1, 1)
        length = inp.size(1)
        length_query = length - self.past
        attn_merges, attn_channels, attn_contexts = [], [], []
        for i in range(self.eval_layers):
            if self.dilated:
                mask = self.mask[i]
            else:
                mask = self.mask
            mask_dec = mask[-length_query:, -length:]
            mask_enc = mask[:length, :length]
            out, attn_merge, attn_channel = self.decoders[i](
                out, context, mask_dec)
            context, attn_context = self.encoders[i](context, mask_enc)
            attn_merges.append(attn_merge)
            attn_channels.append(attn_channel)
            attn_contexts.append(attn_context)
        attn_merge = torch.stack(attn_merges, 1)
        attn_channel = torch.stack(attn_channels, 1)
        attn_context = torch.stack(attn_contexts, 1)
        out = self.linear_out(out)
        out = out.view(-1, length_query, self.future, self.dim) + res
        return out, attn_merge, attn_channel, attn_context


class LinearTemporal(nn.Module):
    def __init__(self, args, adj=None):
        super(LinearTemporal, self).__init__()
        self.past = args.past
        self.future = args.future
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.linear = BottleLinear(self.past, self.future)

    def forward(self, inp):
        '''
        inp: batch x length x dim
        out: batch x length - past x dim
        '''
        out = []
        for i in range(inp.size(1) - self.past):
            inp_i = inp[:, i:i + self.past].transpose(1, 2).contiguous()
            out.append(self.linear(inp_i).transpose(1, 2))
        out = torch.stack(out, 1)
        return out


class LinearSpatial(nn.Module):
    def __init__(self, args, adj=None):
        super(LinearSpatial, self).__init__()
        self.past = args.past
        self.future = args.future
        self.input_size = args.input_size
        self.output_size = args.output_size
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


class LinearSpatialTemporal(nn.Module):
    def __init__(self, args, adj=None):
        super(LinearSpatialTemporal, self).__init__()
        self.past = args.past
        self.future = args.future
        self.input_size = args.input_size
        self.output_size = args.output_size
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


class LinearST(nn.Module):
    def __init__(self, args, adj=None):
        super(LinearST, self).__init__()
        self.past = args.past
        self.future = args.future
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.linear = BottleLinear(
            self.input_size * self.past, self.output_size)

    def forward(self, inp):
        batch, length, dim = inp.size()
        out = []
        for i in range(length - self.past):
            inp_i = inp[:, i:i + self.past].contiguous().view(batch, -1)
            out.append(self.linear(inp_i))
        out = torch.stack(out, 1).view(batch, -1, self.future, dim)
        return out
