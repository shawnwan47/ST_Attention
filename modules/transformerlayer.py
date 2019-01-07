import torch.nn as nn
from modules import MultiHeadedAttention, ResMLP


class TransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(model_dim, heads, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(dropout)
        self.mlp = ResMLP(model_dim, dropout)

    def forward(self, input, bank, mask):
        query = self.layer_norm(input)
        context = self.attn(query, bank, mask)
        output = input + self.drop(context)
        return self.mlp(output)


class SpatialTransformerLayer(TransformerLayer):
    pass


class TemporalTransformerLayer(TransformerLayer):
    def forward(self, input, bank, mask):
        input_t, bank_t = input.transpose(1, 2), bank.transpose(1, 2)
        output = super().forward(input_t, bank_t)
        return output.transpose(1, 2)



class STTransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.layer_t = TemporalTransformerLayer(model_dim, heads, dropout)
        self.layer_s = SpatialTransformerLayer(model_dim, heads, dropout)

    def forward(self, input, bank, mask=None):
        input_s = self.layer_t(input, bank)
        return self.layer_s(input_s, input_s, mask)


class STTransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.layer_bank = STTransformerLayer(model_dim, heads, dropout)
        self.layer_self = STTransformerLayer(model_dim, heads, dropout)

    def forward(self, input, bank, mask):
        input_self = self.layer_bank(input, bank, mask)
        return self.layer_self(input_self, input_self, mask)
