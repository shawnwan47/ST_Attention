import torch
import torch.nn as nn
from torch.nn import functional as F


class GCRNNCell(nn.Module):
    def __init__(self, rnn_type, size, dropout, gc_func, gc_kwargs):
        assert rnn_type in ['RNN', 'GRU']
        super().__init__()
        self.rnn_type = rnn_type
        gate_size = size
        if self.rnn_type == 'RNN': self.activation = nn.ReLU(inplace=True)
        elif self.rnn_type == 'RNNReLU': self.activation = nn.Tanh()
        elif self.rnn_type == 'GRU': gate_size *= 3
        self.gc_i = gc_func(input_size=size, output_size=gate_size, **gc_kwargs)
        self.gc_h = gc_func(input_size=size, output_size=gate_size, **gc_kwargs)
        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, input, hidden):
        if self.rnn_type in ['RNN', 'RNNReLU']:
            output = self.activation(self.gc_i(input) + self.gc_h(hidden))
        elif self.rnn_type == 'GRU':
            output = self._gru(input, hidden)
        output = self.dropout(self.layer_norm(output))
        return output

    def _gru(self, input, hidden):
        i_r, i_i, i_n = self.gc_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.gc_h(hidden).chunk(chunks=3, dim=-1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output


class GCRNN(nn.Module):
    def __init__(self, rnn_type, size, num_layers, dropout, gc_func, gc_kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_nodes = num_nodes
        self.size = size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GCRNNCell(rnn_type, num_nodes, size, dropout, gc_func, gc_kwargs)
            for i in range(num_layers)])

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        shape = (batch_size, self.num_layers, self.num_nodes, self.size)
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
            output_i, hidden = self._forward_tick(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden

    def _forward_tick(self, output, hidden):
        for ilay, layer in enumerate(self.layers):
            hidden[:, ilay] = layer(output, hidden[:, ilay])
            output = hidden[:, ilay]
        return output, hidden


class GARNNCell(GCRNNCell):
    def forward(self, input, hidden):
        if self.rnn_type == 'RNN':
            output_i, attn_i = self.gc_i(input)
            output_h, attn_h = self.gc_h(hidden)
            output = F.tanh(output_i + output_h)
        elif self.rnn_type == 'GRU':
            output, attn_i, attn_h = self._gru(input, hidden)
        output = self.layer_norm(output)
        return output, attn_i, attn_h

    def _gru(self, input, hidden):
        output_i, attn_i = self.gc_i(input)
        output_h, attn_h = self.gc_h(input)
        i_r, i_i, i_n = output_i.chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = output_h.chunk(chunks=3, dim=-1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output, attn_i, attn_h


class GARNN(GCRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.layers = nn.ModuleList([
            GARNNCell(rnn_type, num_nodes, size, gc_func, gc_kwargs)
            for i in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size(0), input.size(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        output, attn_input, attn_hidden = [], [], []
        for idx in range(seq_len):
            output_i, hidden, attn_i, attn_h = self._forward_tick(input[:, idx], hidden)
            output.append(output_i)
            attn_input.append(attn_i)
            attn_hidden.append(attn_h)
        output = torch.stack(output, 1)
        attn_input = torch.stack(attn_input, 1)
        attn_hidden = torch.stack(attn_hidden, 1)
        return output, hidden, attn_input, attn_hidden

    def _forward_tick(self, output, hidden):
        attn_input, attn_hidden = [], []
        for ilay, layer in enumerate(self.layers):
            hidden[:, ilay], attn_i, attn_h = layer(output, hidden[:, ilay])
            attn_input.append(attn_i)
            attn_hidden.append(attn_h)
            output = hidden[:, ilay]
            if ilay < self.num_layers - 1:
                output = self.dropout(output)
        attn_input = torch.stack(attn_input, 1)
        attn_hidden = torch.stack(attn_hidden, 1)
        return output, hidden, attn_input, attn_hidden
