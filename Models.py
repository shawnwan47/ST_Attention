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
        self.hidden_size = args.hidden_size
        self.num_day = args.num_day
        self.num_time = args.num_time
        self.num_loc = args.num_loc // 2
        self.emb_size = args.emb_size
        self.emb_all = self.emb_size * 3
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, self.emb_size),
            nn.Embedding(args.num_time, self.emb_size),
            nn.Embedding(args.num_loc, self.emb_size)
        ])
        self.dropout = nn.Dropout(args.dropout)
        self.linear_in = BottleLinear(self.emb_all + self.past, self.hidden_size)

    def embed(self, features_numerical, features_categorical):
        '''features_categorical: batch x -1 x 4'''
        embeds = [self.dropout(self.embeddings[i](features_categorical[:, :, i]))
                  for i in range(3)]
        features = torch.cat([features_numerical] + embeds, -1)
        return self.linear_in(features)

    def forward_in(self, query_numerical, query_categorical, context_numerical, context_categorical):
        query = self.embed(query_numerical, query_categorical)
        context = self.embed(context_numerical, context_categorical)
        return query, context

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

class MultiHeadedAttention(ModelBase):
    def __init__(self, args):
        super(MultiHeadedAttention, self).__init__(args)
        self.layers = nn.ModuleList([Layers.MultiHeadAttentionLayer(
            self.hidden_size, args.head, args.dropout
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
