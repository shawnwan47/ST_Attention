import torch
import torch.nn as nn

from models import GCRNNCell


class GCRNN(nn.Module):
    def __init__(self, rnn_type, num_nodes, size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_nodes = num_nodes
        self.size = size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GCRNNCell.GCRNNCell(rnn_type, size, gc_func, gc_kwargs)
            for i in range(num_layers)
        ])

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


class GCRNNDecoder(GCRNN):
    def __init__(self, rnn_type, num_nodes, size, out_size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.linear_out = nn.Linear(size, out_size)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        output = self.linear_out(output)
        return output, hidden


class GARNN(GCRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.layers = nn.ModuleList([
            GCRNNCell.GARNNCell(rnn_type, size, gc_func, gc_kwargs)
            for i in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size(0), input.size(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        output = []
        for idx in range(seq_len):
            output_i, hidden, attn_i, attn_h = self._forward_tick(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden, attn_i, attn_h

    def _forward_tick(self, output, hidden):
        attn_input, attn_hidden = [], []
        for ilay, layer in enumerate(self.layers):
            hidden[:, ilay], attn_i, attn_h = layer(output, hidden[:, ilay])
            output = hidden[:, ilay]
            attn_input.append(attn_i)
            attn_hidden.append(attn_h)
        attn_input = torch.stack(attn_input, 1)
        attn_hidden = torch.stack(attn_hidden, 1)
        return output, hidden, attn_input, attn_hidden


class GARNNDecoder(GARNN):
    def __init__(self, rnn_type, num_nodes, size, out_size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.linear_out = nn.Linear(size, out_size)

    def forward(self, input, hidden):
        output, hidden, attn_i, attn_h = super().forward(input, hidden)
        output = self.linear_out(output)
        return output, hidden, attn_i, attn_h
