import torch
import torch.nn as nn

import Layers

from Utils import _get_mask, _get_mask_dilated
from UtilClass import BottleLinear, BottleSparseLinear


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
        self.register_buffer('mask', _get_mask(args.input_length, self.past))

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
        self.dilation = args.dilation
        if self.dilated:
            masks = []
            for i in range(self.num_layers):
                dilation = self.dilation[i]
                window = self.dilation[i + 1] // dilation
                mask = _get_mask_dilated(args.input_length, dilation, window)
                masks.append(mask)
            self.register_buffer('mask', torch.stack(masks, 0))
        else:
            mask = _get_mask(args.input_length, self.past)
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
        self.encoder = Layers.MultiChannelSelfAttention(self.dim, self.channel, adj, args.dropout)
        self.decoder = Layers.MultiChannelContextAttention(self.dim, self.channel, adj, args.dropout)
        self.dilated = args.dilated
        self.dilation = args.dilation
        masks = []
        if self.dilated:
            for i in range(self.num_layers):
                dilation = self.dilation[i]
                window = self.dilation[i + 1] // dilation
                mask = _get_mask_dilated(args.input_length, dilation, window)
                masks.append(mask)
            self.register_buffer('mask', torch.stack(masks, 0))
        else:
            mask = _get_mask(args.input_length, self.past)
            self.register_buffer('mask', mask)

    def forward(self, inp):
        '''
        inp: batch x length x dim
        out: batch x length - past x future x dim
        attn_merge: batch x num_layers x length x channel
        attn_channel: batch x num_layers x channel x length - past x length
        attn_context: batch x num_layers x channel x length x length
        '''
        res = inp[:, self.past:, :self.dim]
        out = inp[:, self.past:]
        context = inp.unsqueeze(1).repeat(1, self.channel, 1, 1)
        length = inp.size(1)
        length_query = length - self.past
        attn_merges, attn_channels, attn_contexts = [], [], []
        for i in range(self.num_layers):
            if self.dilated:
                mask = self.mask[i]
            else:
                mask = self.mask
            mask_decode = mask[-length_query:, -length:]
            mask_encode = mask[:length, :length]
            out, attn_merge, attn_channel = self.decoder(out, context, mask_decode)
            context, attn_context = self.encoder(context, mask_encode)
            attn_merges.append(attn_merge)
            attn_channels.append(attn_channel)
            attn_contexts.append(attn_context)
        attn_merge = torch.stack(attn_merges, 1)
        attn_channel = torch.stack(attn_channels, 1)
        attn_context = torch.stack(attn_contexts, 1)
        return out, attn_merge, attn_channel, attn_context
