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

    def embed(self, inp):
        '''inp: batch x -1 x 4'''
        flow = self.dropout(self.embedding_flow(inp[:, :, 0]))
        day = self.dropout(self.embedding_day(inp[:, :, 1]))
        time = self.dropout(self.embedding_time(inp[:, :, 2]))
        loc = self.dropout(self.embedding_loc(inp[:, :, 3]))
        out = torch.cat((flow, day, time, loc), -1)
        return out


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


class Transformer(ModelBase):
    def __init__(self, args):
        super(Transformer, self).__init__(args)
        self.head = args.head
        self.layers = nn.ModuleList([Layers.TransformerLayer(
            self.emb_size, args.head, args.dropout
        ) for _ in range(self.num_layers)])
        self.linear = BottleLinear(self.emb_size, self.num_flow)

    def forward(self, inp, st):
        '''
        inp: batch x time x loc
        st: batch x time x loc x 3
        out: batch x num_flow x time x loc
        '''
        query, context = self.embed_query_context(inp, st)
        atts = []
        for i in range(self.num_layers):
            query, att = self.layers[i](query, context)
            atts.append(att.cpu())
        out = self.linear(query)
        return self.forward_out(out, atts)

    def forward_out(self, out, att):
        out = out.transpose(1, 2).contiguous()
        out = out.view(-1, self.num_flow, self.future, self.num_loc)
        att = torch.stack(att, 1).view(-1, self.num_layers, self.head,
                                       self.future, self.num_loc,
                                       self.past, self.num_loc)
        return out, att

    def embed_query_context(self, inp, st):
        context = torch.cat((inp.unsqueeze(-1), st), -1)
        query = self.init_query(context)
        context = context.view(-1, self.past * self.num_loc, 4)
        query = query.view(-1, self.future * self.num_loc, 4)
        return self.embed(query), self.embed(context)

    def init_query(self, context):
        query = torch.zeros((context.size(0), self.future, self.num_loc, 4))
        query = query.type(torch.LongTensor)
        for i in range(self.future):
            query[:, i, :, :] = context.data[:, -1, :, :]
            query[:, i, :, 2] += i + 1
            query[:, i, :, 1] += query[:, i, :, 2].div(self.num_time)
            query[:, i, :, 1] = query[:, i, :, 1].remainder(self.num_day)
            query[:, i, :, 2] = query[:, i, :, 2].remainder(self.num_time)
        return Variable(query, volatile=context.volatile).cuda()


class Transformer2(Transformer):
    def __init__(self, args):
        super(Transformer2, self).__init__(args)
        self.emb_flow = args.emb_flow
        self.layers = nn.ModuleList([Layers.Transformer2Layer(
            self.emb_size, self.emb_flow, args.head, args.dropout
        ) for _ in range(self.num_layers)])
        self.linear = BottleLinear(self.emb_flow, self.num_flow)

    def forward(self, inp, st):
        qry, key = self.embed_query_context(inp, st)
        val = key[:, :, :self.emb_flow].contiguous()
        atts = []
        for i in range(self.num_layers):
            qry, att = self.layers[i](qry, key, val)
            atts.append(att.cpu())
        out = self.linear(qry[:, :, :self.emb_flow].contiguous())
        return self.forward_out(out, atts)
