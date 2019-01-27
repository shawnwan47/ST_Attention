import torch.nn as nn
from modules import MultiHeadedAttention, ResMLP


class TransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(model_dim, heads, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(dropout)
        self.mlp = ResMLP(model_dim, dropout)

    def forward(self, query, bank, mask=None):
        query_norm = self.layer_norm(query)
        context = self.drop(self.attn(query_norm, bank, mask))
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

    def forward(self, input, bank, mask_s=None, mask_t=None):
        input_t, bank_t = input.transpose(1, 2), bank.transpose(1, 2)
        input_s = self.layer_t(input_t, bank_t, mask_t).transpose(1, 2)
        return self.layer_s(input_s, input_s, mask_s)


class STTransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn_bank = STTransformerLayer(model_dim, heads, dropout)
        self.attn_self = STTransformerLayer(model_dim, heads, dropout)

    def forward(self, input, bank, mask_s=None, mask_t=None):
        input_self = self.attn_bank(input, bank, mask_s=mask_s)
        return self.attn_self(input_self, input_self, mask_s=mask_s, mask_t=mask_t)
