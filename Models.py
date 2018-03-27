import numpy as np

import torch
import torch.nn as nn

import Layers
from Utils import get_mask_od, get_mask_graph
from UtilClass import *
from Regularizer import *


class ModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.input_size = args.emb_size * 3 + args.flow_size_in
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, args.emb_size),
            nn.Embedding(args.num_time, args.emb_size),
            nn.Embedding(args.num_loc, args.emb_size)
        ])
        self.dropout = nn.Dropout(args.dropout)
        self.linear_in = BottleLinear(self.input_size, self.hidden_size)
        self.linear_out = BottleLinear(self.hidden_size, args.flow_size_out)
        self.relu = nn.ReLU()
        self.mask_od = get_mask_od(args.num_loc, args.inp)
        self.mask_graph = get_mask_graph(args.dataset)

    def forward(self, data_num, data_cat):
        embeds = [self.dropout(self.embeddings[i](data_cat[:, :, i]))
                  for i in range(3)]
        data = torch.cat([data_num, *embeds], -1)
        data = self.relu(self.linear_in(data))
        return context, query

    def get_embeddings(self):
        out = []
        for embedding in self.embeddings:
            out.append(embedding.weight.data.cpu().numpy())
        return np.stack(out)


class Transformer(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.MultiHeadAttentionLayer(
            self.hidden_size, args.head, args.dropout
        ) for _ in range(self.num_layers)])

    def forward(self, data_num, data_cat):
        context, query = super().forward(data_num, data_cat)
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
