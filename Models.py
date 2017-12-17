import torch
import torch.nn as nn

import Layers

from Utils import _get_mask, _get_mask_dilated
from UtilClass import BottleLinear


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


class Attention(nn.Module):
    """
    The Attention model for Spatial-Temporal traffic forecasting
    """

    def __init__(self, args):
        super(Attention, self).__init__()
        self.past = args.past
        self.future = args.future
        self.dim = args.dim
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.linear_out = BottleLinear(self.input_size, self.output_size)
        self.layers = nn.ModuleList([
            Layers.AttentionLayer(self.input_size, args.dropout)
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
        attn = torch.stack(attns, 0)
        return out, attn
