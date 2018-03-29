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
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, args.emb_size),
            nn.Embedding(args.num_time, args.emb_size),
            nn.Embedding(args.num_loc, args.emb_size)
        ])
        self.dropout = nn.Dropout(args.dropout)
        self.linear_flow = BottleLinear(args.flow_size_in, args.hidden_size)
        self.linear_out = BottleLinear(args.hidden_size, args.flow_size_out)
        self.mask_od = get_mask_od(args.num_loc, args.inp)
        self.mask_graph = get_mask_graph(args.dataset)

    def forward(self, data_num, data_cat):
        embeds = [self.dropout(self.embeddings[i](data_cat[:, :, i]))
                  for i in range(3)]
        data = self.linear_flow(data_num) + torch.sum(embeds)
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
        data = super().forward(data_num, data_cat)
        atts = []
        for i in range(self.num_layers):
            data, att = self.layers[i](data, data, data)
            atts.append(att)
        att = torch.stack(atts, 1) # batch x layer x head x query x context
        out = self.linear_out(data)
        return out, att


class Attention(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.AttentionLayer(
            dim=args.hidden_size,
            map_type=args.map_type,
            att_type=args.att_type,
            res=args.res,
            mlp=args.mlp,
            dropout=args.dropout
        ) for _ in range(args.num_layers)])


class AttentionFusion(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.AttentionLayer(
            dim=args.hidden_size,
            map_type=args.map_type,
            att_type=args.att_type,
            res=False
        ) for _ in range(args.num_layers)])
