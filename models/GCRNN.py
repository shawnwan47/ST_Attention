import torch
import torch.nn as nn
from torch.nn import functional as F


class GCRNNCell(nn.Module):
    def __init__(self, rnn_type, size, func, gc_kwargs):
        assert rnn_type in ['RNN', 'RNNReLU', 'GRU']
        super().__init__()
        self.rnn_type = rnn_type
        self.layer_norm = nn.LayerNorm(size)
        gate_size = size
        if self.rnn_type == 'GRU': gate_size *= 3
        elif self.rnn_type == 'LSTM': gate_size *= 4
        self.gc_i = func(size, gate_size, **gc_kwargs)
        self.gc_h = func(size, gate_size, **gc_kwargs)

    def forward(self, input, hidden):
        input = self.layer_norm(input)
        if self.rnn_type == 'RNN':
            output = F.tanh(self.rnn(input, hidden))
        elif self.rnn_type == 'RNNReLU':
            output = F.relu(self.rnn(input, hidden))
        elif self.rnn_type == 'GRU':
            output = self.gru(input, hidden)
        return output

    def rnn(self, input, hidden):
        return self.gc_i(input) + self.gc_h(hidden)

    def gru(self, input, hidden):
        i_r, i_i, i_n = self.gc_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.gc_h(hidden).chunk(chunks=3, dim=-1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output

    def lstm(self, input, hidden):
        hx, cx = hidden
        gates = self.gc_i(input) + self.gc_h(hx)
        g_i, g_f, g_c, g_o = gates.chunk(chunks=4, dim=-1)

        g_i = F.sigmoid(g_i)
        g_f = F.sigmoid(g_f)
        g_c = F.tanh(g_c)
        g_o = F.sigmoid(g_o)

        cy = (g_f * cx) + (g_i * g_c)
        hy = g_o * F.tanh(cy)
        return hy, cy


class GCRNN(nn.Module):
    def __init__(self, rnn_type, num_node, size, num_layers, dropout,
                 func, gc_kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_node = num_node
        self.size = size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.extend(
            [GCRNNCell(rnn_type, size, func, gc_kwargs)
             for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        shape = (batch_size, self.num_layers, self.num_node, self.size)
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(shape), weight.new_zeros(shape))
        else:
            return weight.new_zeros(shape)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size(0), input.size(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        output = []
        for idx in range(seq_len):
            output_i, hidden = self.forward_step(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden

    def forward_step(self, output, hidden):
        for ilay, layer in enumerate(self.layers):
            hidden[:, ilay] = layer(output, hidden[:, ilay])
            output = hidden[:, ilay]
            if ilay < self.num_layers - 1:
                output = self.dropout(output)
        return output, hidden


class GARNN(GCRNN):
    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size(0), input.size(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        output, attn = [], []
        for idx in range(seq_len):
            output_i, hidden, attn_i = self.forward_step(input[:, idx], hidden)
            output.append(output_i)
            attn.append(attn_i)
        output = torch.stack(output, 1)
        attn = torch.stack(attn, 1)
        return output, hidden, attn

    def forward_step(self, output, hidden):
        attn = []
        for ilay, layer in enumerate(self.layers):
            hidden[:, ilay], attn_i = layer(output, hidden[:, ilay])
            attn.append(attn_i)
            output = hidden[:, ilay]
            if ilay < self.num_layers - 1:
                output = self.dropout(output)
        return output, hidden, torch.stack(attn, 1)
