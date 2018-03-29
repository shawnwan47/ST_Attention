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
        emb_size = args.hidden_size - args.flow_size_in
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, emb_size),
            nn.Embedding(args.num_time, emb_size),
            nn.Embedding(args.num_loc, emb_size)
        ])
        self.linear_out = BottleLinear(args.hidden_size, args.flow_size_out)
        self.dropout = nn.Dropout(args.dropout)
        self.mask_od = get_mask_od(args.num_loc, args.inp)
        self.mask_graph = get_mask_graph(args.dataset)

    def forward(self, data_num, data_cat):
        embeds = torch.stack([self.dropout(embedding(data_cat[:, :, i]))
                              for i, embedding in enumerate(self.embeddings)])
        data = torch.cat([data_num, torch.sum(embeds, 0)], -1)
        return data

    def get_embeddings(self):
        out = []
        for embedding in self.embeddings:
            out.append(embedding.weight.data.cpu().numpy())
        return np.stack(out)


class LoneModel(ModelBase):
    def forward(self, data_num, data_cat):
        data = super().forward(data_num, data_cat)
        return self.linear_out(data), None


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
            atts.append(att.cpu())
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
