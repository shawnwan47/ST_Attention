import torch
import torch.nn as nn

from models import TransformerLayer


def gen_mask_temporal(self, size=24):
    ''' Get an attention mask to avoid using the future info.'''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer.TransformerLayer(size, head_count, dropout)
            for i in range(num_layers)
        ])
        self.register_buffer('mask', gen_mask_temporal())

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


class STTransformer(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer.STTransformerLayer(size, head_count, dropout)
            for i in range(num_layers)
        ])
        self.register_buffer('mask', gen_mask_temporal())

    def forward(self, input, bank=None):
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
            attn_temporal.append(attn_t)
            attn_spatial.append(attn_s)
        context = torch.stack(context, 1)
        attn_temporal = torch.stack(attn_temporal, 1)
        attn_spatial = torch.stack(attn_spatial, 1)
        return query, context, attn_temporal, attn_spatial
