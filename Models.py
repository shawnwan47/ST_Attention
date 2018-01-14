import torch
import torch.nn as nn
from torch.nn import functional as F

import Layers

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
        self.emb_all = self.emb_size * 4
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_flow + 1, self.emb_size),
            nn.Embedding(args.num_day, self.emb_size),
            nn.Embedding(args.num_time, self.emb_size),
            nn.Embedding(args.num_loc, self.emb_size)
        ])
        self.dropout = nn.Dropout(args.dropout)
        # for logistic mixtures prediction, bug exists
        # self.linear = BottleLinear(self.emb_all, 3 * args.num_prob)
        # self.prob = Layers.LogisticMixtures(args.num_prob, self.num_flow)
        self.linear = BottleLinear(self.emb_all, self.num_flow)
        self.logsoftmax = nn.LogSoftmax(-1)

    def embed(self, inp):
        '''inp: batch x -1 x 4'''
        embeds = [self.dropout(self.embeddings[i](inp[:, :, i]))
                  for i in range(4)]
        return torch.cat(embeds, -1)

    def forward_in(self, inp, tgt):
        tgt[:, :, :, 0] = self.num_flow
        inp = inp.view(-1, self.past * self.num_loc, 4)
        tgt = tgt.view(-1, self.future * self.num_loc, 4)
        return self.embed(inp), self.embed(tgt)

    def forward_out(self, out, att):
        '''
        out: batch x len_out x num_flow
        att: batch x num_layers x len_out x len_in
        '''
        out = self.logsoftmax(self.linear(out))
        out = out.transpose(1, 2).contiguous()
        out = out.view(-1, self.num_flow, self.future, self.num_loc)
        att = torch.stack(att, 1).view(-1, self.num_layers,
                                       self.future, self.num_loc,
                                       self.past, self.num_loc)
        return out, att

    def get_params(self):
        out = []
        for embedding in self.embeddings:
            out.append(embedding.weight.data.cpu().numpy())
        return out

class Transformer(ModelBase):
    def __init__(self, args):
        super(Transformer, self).__init__(args)
        self.layers = nn.ModuleList([Layers.MultiHeadAttentionLayer(
            self.emb_all, args.head, args.dropout
        ) for _ in range(self.num_layers)])

    def forward(self, inp, tgt):
        '''
        inp: batch x time x loc x 4
        tgt: batch x time x loc x 4
        out: batch x num_flow x time x loc
        '''
        context, query = self.forward_in(inp, tgt)
        atts = []
        for i in range(self.num_layers):
            query, att = self.layers[i](query, context, context)
            atts.append(att)
        return self.forward_out(query, atts)


class Attention(Transformer):
    def __init__(self, args):
        super(Attention, self).__init__(args)
        self.layers = nn.ModuleList([Layers.AttentionLayer(
            dim=self.emb_all,
            map_type=args.map_type,
            att_type=args.att_type,
            res=args.res,
            mlp=args.mlp,
            dropout=args.dropout
        ) for _ in range(self.num_layers)])
