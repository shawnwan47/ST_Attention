import torch
import torch.nn as nn
from torch.nn import functional as F

import Layers
import Attention

import Utils
from UtilClass import *
from Regularizer import *


class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()
        self.past = args.past
        self.future = args.future
        self.num_layers = args.num_layers
        self.num_flow = args.num_flow
        self.num_day = args.num_day
        self.num_time = args.num_time
        self.num_loc = args.num_loc
        self.emb_size = args.emb_size
        self.embedding_flow = nn.Embedding(args.num_flow, args.emb_flow)
        self.embedding_day = nn.Embedding(args.num_day, args.emb_day)
        self.embedding_time = nn.Embedding(args.num_time, args.emb_time)
        self.embedding_loc = nn.Embedding(args.num_loc, args.emb_loc)
        self.dropout = nn.Dropout(args.dropout)
        self.softmax = nn.Softmax(3)

    def embed(self, inp):
        '''inp: batch x time x loc x 4'''
        times = inp.size(1)
        inp = inp.view(-1, times * self.num_loc, 4)
        flow = self.dropout(self.embedding_flow(inp[:, :, 0]))
        day = self.dropout(self.embedding_day(inp[:, :, 1]))
        time = self.dropout(self.embedding_time(inp[:, :, 2]))
        loc = self.dropout(self.embedding_loc(inp[:, :, 3]))
        out = torch.cat((flow, day, time, loc), -1)
        return out.view(-1, times, self.num_loc, self.emb_size)


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


class RNN(ModelBase):
    def __init__(self, args):
        super(RNN, self).__init__(args)
        self.rnn = Layers.RNN(
            args.rnn_type, self.emb_size, self.emb_size,
            args.num_layers, args.dropout)
        self.linear_out = BottleLinear(args.hidden_size, args.num_flow)

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
        self.att = Attention.Attn(self.hidden_size, args.att_type, args.dropout)
        mask = Utils.get_mask_trim(args.max_len, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        out = self.forward_rnn(inp, daytime)
        batch, length, dim = out.size()
        mask = self.mask[:length, :length]
        out, att = self.att(out, mask)
        out = self.linear_out(self.dropout(out[:, self.past:]))
        out = out.view(batch, -1, self.future, self.dim)
        return out, att


class HeadAttn(ModelBase):
    def __init__(self, args):
        super(HeadAttn, self).__init__(args)
        self.head = args.head
        self.layers = nn.ModuleList([Layers.HeadAttnLayer(
            self.input_size, self.head, args.dropout)
            for _ in range(self.num_layers)])
        self.linear_out = BottleLinear(self.input_size, self.dim * self.future)
        mask = Utils.get_mask_trim(args.max_len, self.past)
        self.register_buffer('mask', mask)

    def forward(self, inp, daytime=None):
        out = super(HeadAttn, self).forward(inp, daytime)
        batch, length, dim = out.size()
        att = []
        for i in range(self.num_layers):
            out, att_i = self.layers[i](out, self.mask[:length, :length])
            att.append(att_i)
        att = torch.stack(att, 1)
        out = out[:, self.past:].contiguous()
        out = self.linear_out(out).view(batch, -1, self.future, self.dim)
        out = out + inp[:, self.past:].unsqueeze(-2)
        return out, att

    def pack_weight(self, weight):
        return weight.sum(1).view(-1, weight.size(-1))


class ST_Transformer(ModelBase):
    def __init__(self, args):
        super(ST_Transformer, self).__init__(args)
        self.layers = nn.ModuleList([Layers.TransformerLayer(
            self.emb_size, self.emb_size, args.head, args.dropout
        ) for _ in range(self.num_layers)])
        self.linear = BottleLinear(self.emb_size, self.num_flow, bias=False)

    def forward(self, inp, st):
        '''
        inp: batch x time x loc
        st: batch x time x loc x 3
        out: batch x num_flow x time x loc
        '''
        key = torch.cat((inp.unsqueeze(-1), st), -1)
        out = self.init_out(key)
        key = self.embed(key)
        out = self.embed(out)
        # flatten inp key
        key = key.view(-1, self.past * self.num_loc, self.emb_size)
        out = out.view(-1, self.future * self.num_loc, self.emb_size)
        for i in range(self.num_layers):
            out, att = self.layers[i](out, key, key)
        out = out.view(-1, self.future, self.num_loc, self.emb_size)
        out = self.linear(out).view(-1, self.num_flow)
        att = att.view(-1, self.future, self.num_loc, self.past, self.num_loc)
        att = att[:, 0]
        return out, att

    def init_out(self, key):
        out = torch.zeros((key.size(0), self.future, self.num_loc, 4)).type(torch.LongTensor)
        out[:, :, :, 1:] = key[:, -1, :, 1:].data
        for i in range(self.future):
            out[:, i, :, 2] = out[:, i, :, 2] + i
            out[:, i, :, 1] = out[:, i, :, 1] + out[:, -1, :, 2].div(self.num_time)
            out[:, i, :, 1] = out[:, i, :, 1].remainder(self.num_day)
            out[:, i, :, 2] = out[:, i, :, 2].remainder(self.num_time)
        return Variable(out, volatile=key.volatile).cuda()
