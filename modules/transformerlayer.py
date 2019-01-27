import torch.nn as nn
from modules import MultiHeadedAttention, ResMLP


class TransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(model_dim, heads, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(dropout)
        self.mlp = ResMLP(model_dim, dropout)

    def forward(self, query, bank=None):
        query_norm = self.layer_norm(query)
        bank = query_norm if bank is None else query_norm
        context = self.drop(self.attn(query_norm, bank))
        return self.mlp(query + context)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn_bank = TransformerLayer(model_dim, heads, dropout)
        self.attn_self = TransformerLayer(model_dim, heads, dropout)

    def forward(self, input, bank, mask=None):
        input_self = self.attn_bank(input, bank)
        return self.attn_self(input_self, input_self, mask)


class STTransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.layer_t = TransformerLayer(model_dim, heads, dropout)
        self.layer_s = TransformerLayer(model_dim, heads, dropout)

    def forward(self, query, bank=None):
        query_t = query.transpose(1, 2)
        bank_t = bank if bank is None else bank.transpose(1, 2)
        query_s = self.layer_t(query_t, bank_t).transpose(1, 2)
        return self.layer_s(query_s)


class STTransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.layer_bank = STTransformerLayer(model_dim, heads, dropout)
        self.layer_self = STTransformerLayer(model_dim, heads, dropout)

    def forward(self, input, bank):
        input_self = self.layer_bank(input, bank)
        return self.layer_self(input_self)
