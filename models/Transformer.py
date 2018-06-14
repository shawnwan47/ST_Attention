import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_temporal(self, size=24):
    ''' Get an attention mask to avoid using the future info.'''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask

class PositionwiseFeedForward(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, size)
        self.relu = nn.ReLU(inplace=True)
        self.w_2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input):
        output = self.dropout(self.relu(self.w_1(input)))
        output = self.dropout(self.w_2(output)) + input
        return self.layer_norm(output)


class Transformer(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(size, head_count, dropout)
            for i in range(num_layers)
        ])
        self.register_buffer('mask', mask_temporal())

    def forward(self, input, bank=None):
        len_input = input.size(-2)
        if bank is None:
            mask = self.mask[-len_input:, -len_input:]
        else:
            mask = self.mask[-len_input:, -(len_input + bank.size(-2)):]
        output, attn = [input], []
        for i, layer in enumerate(self.layers):
            if bank is None:
                context = output[-1]
            else:
                context = torch.cat((bank[:, i], output[-1]), -2)
            output_i, attn_i = layer(output[-1], context, mask)
            output.append(output_i)
            attn.append(attn_i)
        return output[-1], torch.stack(output, 1), torch.stack(attn, 1)


class TransformerLayer(nn.Module):
    def __init__(self, size, head_count, dropout):
        super().__init__()
        self.attention = Attention.MultiAttention(size, size, head_count, dropout)
        self.layer_norm = nn.LayerNorm(size)
        self.feed_forward = PositionwiseFeedForward(size, dropout)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, context, mask):
        context, attn = self.attention(context, context, query, mask)
        output = self.dropout(context) + query
        output = self.layer_norm(output)
        output = self.feed_forward(output)
        return output, attn


class STTransformer(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerLayer(size, head_count, dropout)
            for i in range(num_layers)
        ])
        self.register_buffer('mask', mask_temporal())

    def forward(self, input, bank=None):
        len_input = input.size(-3)
        if bank is None:
            mask = self.mask[-len_input:, -len_input:]
        else:
            mask = self.mask[-len_input:, -(len_input + bank.size(-3)):]
        output, attn_temporal, attn_spatial = [input], [], []
        for i, layer in enumerate(self.layers):
            if bank is None:
                context = output[-1]
            else:
                context = torch.cat((bank[:, i], output[-1]), -3)
            output_i, attn_t_i, attn_s_i = layer(output[-1], context, mask)
            output.append(output_i)
            attn_temporal.append(attn_t_i)
            attn_spatial.append(attn_s_i)
        attn_temporal = torch.stack(attn_temporal, 1)
        attn_spatial = torch.stack(attn_spatial, 1)
        return output[-1], torch.stack(output, 1), attn_temporal, attn_spatial


class STTransformerLayer(nn.Module):
    def __init__(self, size, head_count, dropout):
        self.attn_temporal = Attention.MultiAttention(size, head_count, dropout)
        self.attn_spatial = Attention.MultiAttention(size, head_count, dropout)
        self.layer_norm_temporal = nn.LayerNorm(size)
        self.layer_norm_spatial = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, context):
