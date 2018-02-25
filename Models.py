import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import Layers

import Utils
from UtilClass import *
from Regularizer import *


class ModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inp = args.inp
        self.out = args.out
        self.past = args.past
        self.future = args.future
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.input_size = args.emb_size * 3 + self.past
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, args.emb_size),
            nn.Embedding(args.num_time, args.emb_size),
            nn.Embedding(args.num_loc, args.emb_size)
        ])
        self.dropout = nn.Dropout(args.dropout)
        self.linear_in = BottleLinear(self.input_size, self.hidden_size)
        self.linear_out = BottleLinear(self.hidden_size, self.future)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, data_numerical, data_categorical):
        embeds = [self.dropout(self.embeddings[i](data_categorical[:, :, i]))
                  for i in range(3)]
        data = torch.cat([data_numerical, *embeds], -1)
        data = self.tanh(self.linear_in(data))
        context = self.getOD(data, self.inp)
        query = self.getOD(data, self.out)
        return context, query

    def getOD(self, features, od):
        stations = features.size(2) // 2
        if od == 'O':
            return features[:, :, :stations].contiguous()
        elif od == 'D':
            return features[:, :, -stations:].contiguous()
        else:
            return features

    def getParams(self):
        out = []
        for embedding in self.embeddings:
            out.append(embedding.weight.data.cpu().numpy())
        return np.stack(out)

class SpatialAttention(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.MultiHeadAttentionLayer(
            self.hidden_size, args.head, args.dropout
        ) for _ in range(self.num_layers)])

    def forward(self, data_numerical, data_categorical):
        context, query = super().forward(data_numerical, data_categorical)
        atts = []
        for i in range(self.num_layers):
            query, att = self.layers[i](query, context, context)
            atts.append(att)
        att = torch.stack(atts)
        out = self.linear_out(query)
        return out, att


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
