import torch
import torch.nn as nn
from modules import HeadAttendedAttention, MLP, ResMLP


class GAAT(nn.Module):
    def __init__(self, embedding, model_dim, out_dim, num_layers, heads,
                 dropout=0.1, mask=None):
        super().__init__()
        self.embedding = embedding
        self.encoder = GAATEncoder(model_dim, num_layers, heads, dropout, mask)
        self.decoder = MLP(model_dim, out_dim, dropout)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        hidden, attn = self.encoder(input)
        output = self.decoder(hidden)
        return output


class GAATEncoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout, mask):
        super().__init__()
        self.layers = nn.ModuleList([
            GAATLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.register_buffer('mask', mask)

    def forward(self, x):
        attns = []
        for layer in self.layers:
            y, attn = layer(x, mask=self.mask)
            attns.append(attn)
        attn = torch.stack(attns, 1)
        return y


class GAATLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.attn = HeadAttendedAttention(model_dim, heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.fc_q_gate = nn.Linear(model_dim, 1, bias=False)
        self.fc_c_gate = nn.Linear(model_dim, 1)
        self.resmlp = ResMLP(model_dim, dropout)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, query, bank=None, mask=None):
        bank = query if bank is None else bank
        context, attn = self.attn(query, bank, mask)
        gate = torch.sigmoid(self.fc_q_gate(query) + self.fc_c_gate(context))
        fusion = query * gate + self.drop(context) * (1 - gate)
        output = self.ln(self.resmlp(fusion))
        attn.mul_(gate.unsqueeze(1))
        return output
