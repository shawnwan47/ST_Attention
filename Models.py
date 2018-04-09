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
        self.embeddings = nn.ModuleList([
            nn.Embedding(args.num_day, args.day_embed_size),
            nn.Embedding(args.num_time, args.time_embed_size),
            nn.Embedding(args.num_loc, args.loc_embed_size)
        ])
        self.linear_in = nn.Linear(args.input_size, args.hidden_size)
        self.linear_out = nn.Linear(args.hidden_size, args.output_size)
        self.dropout = nn.Dropout(args.dropout)
        if args.mask:
            mask_adj = get_mask_adj(args.dataset)
            mask_od = get_mask_od(args.num_loc, args.input_od)
            self.register_buffer('mask', mask_adj | mask_od)
        else:
            self.mask = None

    def forward(self, data_num, data_cat):
        embeds = (self.dropout(embedding(data_cat[:, :, i]))
                  for i, embedding in enumerate(self.embeddings))
        data = torch.cat((data_num, *embeds), -1)
        return self.dropout(F.relu(self.linear_in(data)))

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
        self.layers = nn.ModuleList([Layers.TransformerLayer(
            args.head, args.hidden_size, args.dropout
        ) for _ in range(args.num_layers)])

    def forward(self, data_num, data_cat):
        hidden = super().forward(data_num, data_cat)
        atts = []
        for layer in self.layers:
            hidden, att = layer(hidden, self.mask)
            atts.append(att.cpu())
        att = torch.stack(atts, 1) # batch x layer x head x query x context
        output = self.linear_out(hidden)
        return output, att


class TransformerGate(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.TransformerGateLayer(
            args.head, args.hidden_size, args.dropout
        ) for _ in range(args.num_layers)])

    def forward(self, data_num, data_cat):
        hidden = super().forward(data_num, data_cat)
        atts, gates = [], []
        for layer in self.layers:
            hidden, att, gate = layer(hidden, self.mask)
            atts += att,
            gates += gate,
        output = self.linear_out(hidden)
        att = torch.stack(atts, 1)
        gate = torch.stack(gates, 1)
        return output, att, gate


class TransformerFusion(TransformerGate):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList([Layers.TransformerFusionLayer(
            args.head, args.hidden_size, args.dropout
        ) for _ in range(args.num_layers)])
