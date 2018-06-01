import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, size, num_layers, head_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(size, head_count, dropout)
            for i in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input, hidden=None):
        out = self.layers[0](input, hidden[:, 0])
        for i, layer in enumerate(self.layers[1:]):
            out = layer(out, hidden[:, i])
        out = self.layer_norm(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, size, dropout, head_count=8, hidden_size=2048):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm = onmt.modules.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)



class PositionwiseFeedForward(nn.Module):
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = nn.LayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x
