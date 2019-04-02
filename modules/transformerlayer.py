import torch.nn as nn
from modules import MultiHeadedAttention, ResMLP


class TransformerLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(model_dim, heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.resmlp = ResMLP(model_dim, dropout)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, query, bank=None):
        bank = query if bank is None else bank
        context, attn = self.attn(query, bank)
        output = self.resmlp(query + self.drop(context))
        return self.ln(output), attn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.layer_bank = TransformerLayer(model_dim, heads, dropout)
        self.layer_self = TransformerLayer(model_dim, heads, dropout)

    def forward(self, input, bank):
        input_self, attn_bank = self.layer_bank(input, bank)
        output, attn_self = self.layer_self(input_self)
        return output


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
