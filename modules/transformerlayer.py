import torch.nn as nn
from modules import MultiHeadedAttention, ResMLP


class TransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(model_dim, heads, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(dropout)
        self.mlp = ResMLP(model_dim, dropout)

    def forward(self, input, bank, mask=None):
        query = self.layer_norm(input)
        output = self.drop(self.attn(query, bank, mask))
        return self.mlp(input + output)


class STransformerLayer(TransformerLayer):
    pass


class TTransformerLayer(TransformerLayer):
    def forward(self, input, bank):
        input_t = input.transpose(1, 2).contiguous()
        bank_t = bank.transpose(1, 2).contiguous()
        output = super().forward(input_t, bank_t)
        return output.transpose(1, 2)


class STTransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.layer_t = TTransformerLayer(model_dim, heads, dropout)
        self.layer_s = STransformerLayer(model_dim, heads, dropout)

    def forward(self, input, bank, mask):
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
