import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import Layers
from Utils import getOD
from UtilClass import *
from Regularizer import *


class ModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inp = args.inp
        self.out = args.out
        self.flow_size_in = args.flow_size_in
        self.flow_size_out = args.flow_size_out
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.input_size = args.emb_size * 3 + self.flow_size_in
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, args.emb_size),
            nn.Embedding(args.num_time, args.emb_size),
            nn.Embedding(args.num_loc, args.emb_size)
        ])
        self.dropout = nn.Dropout(args.dropout)
        self.linear_in = BottleLinear(self.input_size, self.hidden_size)
        self.linear_out = BottleLinear(self.hidden_size, self.flow_size_out)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, data_numerical, data_categorical):
        embeds = [self.dropout(self.embeddings[i](data_categorical[:, :, i]))
                  for i in range(3)]
        data = torch.cat([data_numerical, *embeds], -1)
        data = self.tanh(self.linear_in(data))
        context = getOD(data, self.inp)
        query = getOD(data, self.out)
        return context, query

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
        att = torch.stack(atts, 1) # batch x layer x head x query x context
        out = self.linear_out(query)
        return out, att


class Attention(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.AttentionLayer(
            dim=self.emb_all,
            map_type=args.map_type,
            att_type=args.att_type,
            res=args.res,
            mlp=args.mlp,
            dropout=args.dropout
        ) for _ in range(self.num_layers)])
