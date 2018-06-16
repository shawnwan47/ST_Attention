import numpy as np
import torch
import torch.nn as nn

from models import TransformerLayer


def gen_temporal_mask(size=24):
    ''' Get an attention mask to avoid using the future info.'''
    subsequent_mask = np.triu(np.ones((size, size)), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


def gen_temporal_distance(size=24):
    dist = np.arange(size).reshape(-1, 1) + np.arange(0, -size, -1)
    dist = torch.from_numpy(np.tril(dist))
    return dist


class Transformer(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer.TransformerLayer(size, head_count, dropout)
            for i in range(num_layers)
        ])
        self.register_buffer('mask', gen_temporal_mask())

    def forward(self, input, bank=None):
        len_input = input.size(-2)
        if bank is None:
            mask = self.mask[-len_input:, -len_input:]
        else:
            mask = self.mask[-len_input:, -(len_input + bank.size(-2)):]
        query, context, attn = input, [], []
        for i, layer in enumerate(self.layers):
            if bank is None:
                context.append(query)
            else:
                context.append(torch.cat((bank[:, i], query), -2))
            query, attn_i = layer(query, context[-1], mask)
            attn.append(attn_i)
        context = torch.stack(context, 1)
        attn = torch.stack(attn)
        return query, context, attn


class RelativeTransformer(Transformer):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__(size, num_layers, head_count, dropout)
        temporal_dist = gen_temporal_distance()
        self.layers = nn.ModuleList([
            TransformerLayer.RelativeTransformerLayer(
                size, head_count, dropout, temporal_dist)
            for i in range(num_layers)
        ])


class STTransformer(Transformer):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__(size, num_layers, head_count, dropout)
        self.layers = nn.ModuleList([
            TransformerLayer.STTransformerLayer(size, head_count, dropout)
            for i in range(num_layers)
        ])

    def forward(self, input, bank=None):
        '''
        output: batch x lenq x node x size
        attn_s: batch x lenq x node x node
        attn_spatial: batch x lenq x lay x node x node
        attn_t: batch x node x lenq x lenc
        attn_temporal: batch x lenq x lay x node x lenc
        '''
        len_input = input.size(-3)
        if bank is None:
            mask = self.mask[-len_input:, -len_input:]
        else:
            mask = self.mask[-len_input:, -(len_input + bank.size(-3)):]
        query, context, attn_temporal, attn_spatial = input, [], [], []
        for i, layer in enumerate(self.layers):
            if bank is None:
                context.append(query)
            else:
                context.append(torch.cat((bank[:, i], query), -3))
            query, attn_t, attn_s = layer(query, context[-1], mask)
            attn_spatial.append(attn_s)
            attn_temporal.append(attn_t.transpose(-2, -3))
        context = torch.stack(context, 1)
        attn_spatial = torch.stack(attn_spatial, 1)
        attn_temporal = torch.stack(attn_temporal, 2)
        return query, context, attn_temporal, attn_spatial


class RelativeSTTransformer(STTransformer):
    def __init__(self, size, num_layers, head_count, dropout, spatial_dist):
        super().__init__(size, num_layers, head_count, dropout)
        temporal_dist = gen_temporal_distance()
        self.layers = nn.ModuleList([
            TransformerLayer.RelativeSTTransformerLayer(
                size, head_count, dropout, temporal_dist, spatial_dist)
            for i in range(num_layers)
        ])
