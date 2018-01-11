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
        self.num_loc = args.num_loc // 2
        self.emb_size = args.emb_size
        self.emb_merge = args.emb_merge
        self.emb_all = args.emb_all
        self.embedding_flow = nn.Embedding(args.num_flow + 1, self.emb_size)
        self.embedding_day = nn.Embedding(args.num_day, self.emb_size)
        self.embedding_time = nn.Embedding(args.num_time, self.emb_size)
        self.embedding_loc = nn.Embedding(args.num_loc, self.emb_size)
        self.dropout = nn.Dropout(args.dropout)

    def embed(self, inp):
        '''inp: batch x -1 x 4'''
        flow = self.dropout(self.embedding_flow(inp[:, :, 0]))
        day = self.dropout(self.embedding_day(inp[:, :, 1]))
        time = self.dropout(self.embedding_time(inp[:, :, 2]))
        loc = self.dropout(self.embedding_loc(inp[:, :, 3]))
        emb = (flow, day, time, loc)
        if self.emb_merge == 'cat':
            out = torch.cat(emb, -1)
        else:
            out = torch.sum(torch.stack(emb, 0), 0)
        return out


class Transformer(ModelBase):
    def __init__(self, args):
        super(Transformer, self).__init__(args)
        self.head = args.head
        self.layers = nn.ModuleList([Layers.TransformerLayer(
            self.emb_all, args.head, args.dropout
        ) for _ in range(self.num_layers)])
        self.linear = BottleLinear(self.emb_all, self.num_flow)

    def embed_context_query(self, inp, tgt):
        tgt[:, :, :, 0] = self.num_flow
        inp = inp.view(-1, self.past * self.num_loc, 4)
        tgt = tgt.view(-1, self.future * self.num_loc, 4)
        return self.embed(inp), self.embed(tgt)

    def forward(self, inp, tgt):
        '''
        inp: batch x time x loc x 4
        tgt: batch x time x loc x 4
        out: batch x num_flow x time x loc
        '''
        context, query = self.embed_context_query(inp, tgt)
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


class TransformerSimple(Transformer):
    def __init__(self, args):
        super(TransformerSimple, self).__init__(args)
        self.layers = nn.ModuleList([Layers.TransformerLayer(
            self.emb_all, args.head, args.dropout, simple=True
        ) for _ in range(self.num_layers)])
