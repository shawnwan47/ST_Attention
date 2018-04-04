import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import Layers
from Utils import get_mask_od, get_mask_adj
from UtilClass import *
from Regularizer import *


class ModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, args.day_embed_size),
            nn.Embedding(args.num_time, args.time_embed_size),
            nn.Embedding(args.num_loc, args.loc_embed_size)
        ])
        self.linear_in = BottleLinear(args.input_size, args.hidden_size)
        self.linear_out = BottleLinear(args.hidden_size, args.output_size)
        self.dropout = nn.Dropout(args.dropout)
        mask_adj = get_mask_adj(args.dataset)
        mask_od = get_mask_od(args.num_loc, args.inp)
        self.mask = mask_adj | mask_od

    def forward(self, data_num, data_cat):
        embeds = (self.dropout(embedding(data_cat[:, :, i]))
                  for i, embedding in enumerate(self.embeddings))
        data = torch.cat((data_num, *embeds), -1)
        hid = self.dropout(F.relu(self.linear_in(data)))
        return hid

    def get_embeddings(self):
        return np.stack([embedding.weight.data.cpu().numpy()
                         for embedding in self.embeddings])


class Isolation(ModelBase):
    def forward(self, data_num, data_cat):
        data = super().forward(data_num, data_cat)
        return self.linear_out(data)


class Transformer(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.MultiHeadAttentionLayer(
            args.hidden_size, args.head, args.dropout
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


class AttentionFusion(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.AttentionFusionLayer(
            dim=args.hidden_size,
            head=args.head,
            att_type=args.att_type,
            dropout=args.dropout
        ) for _ in range(args.num_layers)])

    def forward(self, data_num, data_cat):
        data = super().forward(data_num, data_cat)
        gates, att_fusions, att_heads = [], [], []
        for layer in self.layers:
            data, gate, att_fusion, att_head = layer(data, data, self.mask)
            gates.append(gate)
            att_fusions.append(att_fusion)
            att_heads.append(att_head)
        gates = torch.stack(gates, 1)
        att_fusions = torch.stack(att_fusions, 1)
        att_heads = torch.stack(att_heads, 1)
        out = self.linear_out(data)
        return out, gates, att_fusions, att_heads
