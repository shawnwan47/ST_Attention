import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGRU(nn.Module):
    def __init__(self, model_dim, num_layers, dropout, func, kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GraphGRUCell(model_dim, func, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        if hidden == None:
            hidden = self.init_hidden()
        output = []
        for idx in range(input.size(1)):
            output_i, hidden = self.forward_i(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden

    def forward_i(self, input, hidden):
        hidden_new = []
        for ilay, layer in enumerate(self.layers):
            input = input + self.drop(layer(input, hidden[:, ilay]))
            hidden_new.append(input)
        hidden_new = torch.stack(hidden_new, 1)
        return input, hidden_new

    def init_hidden(self):
        shape = (1, self.num_layers, 1, self.model_dim)
        return next(self.parameters()).new_zeros(shape)


class GraphGRUCell(nn.Module):
    def __init__(self, model_dim, func, kwargs):
        super().__init__()
        self.func_i = func(model_dim, model_dim * 3, **kwargs)
        self.func_h = func(model_dim, model_dim * 3, **kwargs)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input, hidden):
        input_norm = self.layer_norm(input)
        return input + self.gru(input_norm, hidden)

    def gru(self, input, hidden):
        i_r, i_i, i_n = self.func_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.func_h(hidden).chunk(chunks=3, dim=-1)
        gate_r = F.sigmoid(i_r + h_r)
        gate_i = F.sigmoid(i_i + h_i)
        i_new = F.tanh(i_n + gate_r * h_n)
        return i_new + gate_i * (hidden - i_new)
