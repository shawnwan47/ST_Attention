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


class TransformerBase(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer.TransformerLayer(size, head_count, dropout)
            for _ in range(num_layers)
        ])
        self.register_buffer('mask', gen_temporal_mask())


class RelativeTransformerBase(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        dist_t = gen_temporal_distance()
        self.layers = nn.ModuleList([
            TransformerLayer.RelativeTransformerLayer(
                size, head_count, dropout, dist_t)
            for _ in range(num_layers)
        ])
        self.register_buffer('mask', gen_temporal_mask())


class STTransformerBase(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers_t = nn.ModuleList([
            TransformerLayer.TransformerLayer(size, head_count, dropout)
            for _ in range(num_layers)
        ])
        self.layers_s = nn.ModuleList([
            TransformerLayer.TransformerLayer(size, head_count, dropout)
            for _ in range(num_layers)
        ])
        self.register_buffer('mask', gen_temporal_mask())


class RelativeSTTransformerBase(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout, dist_s):
        super().__init__()
        self.num_layers = num_layers
        self.layers_s = nn.ModuleList([
            TransformerLayer.RelativeTransformerLayer(
                size, head_count, dropout, dist_s)
            for _ in range(num_layers)
        ])
        dist_t = gen_temporal_distance()
        self.layers_t = nn.ModuleList([
            TransformerLayer.RelativeTransformerLayer(
                size, head_count, dropout, dist_t)
            for _ in range(num_layers)
        ])
        self.register_buffer('mask', gen_temporal_mask())


class TransformerEncoder(TransformerBase):
    def forward(self, input):
        '''
        input, output: batch x lenq x size
        bank, memory: lay x batch x lenm x size
        attn: lay x batch x head x lenq x lenm
        '''
        len_input = input.size(-2)
        mask = self.mask[-len_input:, -len_input:]
        query, memory, attn = input, [], []
        for idx, layer in enumerate(self.layers):
            memory.append(query)
            query, attn_i = layer(query, memory[-1], mask)
            attn.append(attn_i)
        memory = torch.stack(memory)
        attn = torch.stack(attn)
        return query, memory, attn


class TransformerDecoder(TransformerBase):
    def forward(self, input, bank):
        len_input = input.size(-2)
        mask = self.mask[-len_input:, -(len_input + bank.size(-2)):]
        query, memory, attn = input, [], []
        for idx, layer in enumerate(self.layers):
            memory.append(torch.cat((bank[idx], query), -2))
            query, attn_i = layer(query, memory[-1], mask)
            attn.append(attn_i)
        memory = torch.stack(memory)
        attn = torch.stack(attn)
        output = self.linear_out(query)
        return output, memory, attn


class STTransoformerEncoder(STTransformerBase):
    def forward(self, input):
        '''
        input, output: batch x lenq x node x size
        bank: lay x batch x node x lenq x size
        attn_s: batch x head x node x node
        attn_spatial: batch x lay x head x node x node
        attn_t: batch x node x head x lenq x lenm
        attn_temporal: batch x lay x node x head x lenq x lenm
        '''
        len_input = input.size(1)
        mask = self.mask[-len_input:, -len_input:]
        query, memory, attn_temporal, attn_spatial = input, [], [], []
        for idx in range(self.num_layers):
            query, attn_s = self.layers_s[idx](query, query)
            query_t = query.transpose(1, 2)
            memory.append(query_t)
            query_t, attn_t = self.layers_t[idx](query_t, memory[-1], mask)
            query = query_t.transpose(1, 2)
            attn_spatial.append(attn_s)
            attn_temporal.append(attn_t)
        memory = torch.stack(memory)
        attn_spatial = torch.stack(attn_spatial)
        attn_temporal = torch.stack(attn_temporal)
        return query, memory, attn_temporal, attn_spatial



class STTransformerDecoder(STTransformerBase):
    def forward(self, input, bank):
        '''
        input, output: batch x lenq x node x size
        bank: lay x batch x node x lenq x size
        attn_s: batch x lenq x head x node x node
        attn_spatial: batch x lay x lenq x head x node x node
        attn_t: batch x node x head x lenq x lenm
        attn_temporal: batch x lay x node x head x lenq x lenm
        '''
        len_input = input.size(1)
        mask = self.mask[-len_input:, -(len_input + bank.size(-2)):]
        query, memory, attn_temporal, attn_spatial = input, [], [], []
        for idx in range(self.num_layers):
            query, attn_s = self.layers_s[idx](query, query)
            query_t = query.transpose(1, 2)
            memory.append(torch.cat((bank[idx], query_t), -2))
            query_t, attn_t = self.layers_t[idx](query_t, memory[-1], mask)
            query = query_t.transpose(1, 2)
            attn_spatial.append(attn_s)
            attn_temporal.append(attn_t)
        memory = torch.stack(memory)
        attn_spatial = torch.stack(attn_spatial)
        attn_temporal = torch.stack(attn_temporal)
        output = self.linear_out(query)
        return output, memory, attn_temporal, attn_spatial


class RelativeTransformerEncoder(RelativeTransformerBase, TransformerEncoder):
    pass


class RelativeTransformerDecoder(RelativeTransformerBase, TransformerDecoder):
    pass


class RelativeSTTransformerEncoder(RelativeSTTransformerBase, STTransoformerEncoder):
    pass


class RelativeSTTransformerEncoder(RelativeSTTransformerBase, STTransformerDecoder):
    pass
