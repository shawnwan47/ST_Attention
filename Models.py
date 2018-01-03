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
        self.emb_size = args.emb_size
        self.embedding_flow = nn.Embedding(args.num_flow, args.emb_flow)
        self.embedding_day = nn.Embedding(args.num_day, args.emb_day)
        self.embedding_time = nn.Embedding(args.num_time, args.emb_time)
        self.embedding_loc = nn.Embedding(args.num_loc, args.emb_loc)
        self.dropout = nn.Dropout(args.dropout)
        self.softmax = nn.Softmax(3)

    def embed(self, inp):
        '''inp: day x time x loc x 4'''
        flow = self.embedding_flow(inp[:, :, :, 0])
        day = self.embedding_day(inp[:, :, :, 1])
        time = self.embedding_time(inp[:, :, :, 2])
        loc = self.embedding_loc(inp[:, :, :, 3])
        return torch.cat((flow, day, time, loc), -1)


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
        self.attn = Attention.Attn(self.hidden_size, args.attn_type, args.dropout)
        mask = Utils.get_mask_trim(args.max_len, self.past)
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
        mask = Utils.get_mask_trim(args.max_len, self.past)
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
        out = out + inp[:, self.past:].unsqueeze(-2)
        return out, attn

    def pack_weight(self, weight):
        return weight.sum(1).view(-1, weight.size(-1))


class ST_Transformer(ModelBase):
    def __init__(self, args):
        super(ST_Transformer, self).__init__(args)
        self.attn_st = Attention.MixAttn(self.emb_size,
                                         ['general', 'mlp'],
                                         dropout=args.dropout)
        self.register_buffer('mask', Utils.get_mask_trim(args.max_len))

    def forward(self, inp):
        '''
        inp: day, time, loc, 4
        out: batch x length - past x future x dim
        '''
        out = self.embed(inp)
        day, time, loc, dim = out.size()
        tmp_embedding = self.embed_daytime(daytime)
        spa_embedding = self.embed_station(batch, length)
        st_embedding = torch.cat((spa_embedding, tmp_embedding.unsqueeze(-2)), -1)
        # batch x length x length
        attn_tmp = self.attn_temporal(
            tmp_embedding, tmp_embedding, self.mask[:length, :length])
        # dim x dim
        attn_spa = self.attn_spatial(spa_embedding, spa_embedding)
