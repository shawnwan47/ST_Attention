import torch
import torch.nn as nn


class GraphRNN(nn.Module):
    def __init__(self, rnn_type, size, num_layers, num_nodes,
                 func=nn.Linear, **func_args):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_nodes = num_nodes
        self.size = size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GraphRNNCell(rnn_type, size, func, **func_args)
            for i in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size(0), input.size(1)
        if hidden is None:
            hidden = self._init_hidden(batch_size)
        output = []
        for idx in range(seq_len):
            output_i, hidden = self._forward_tick(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden

    def _init_hidden(self, batch_size):
        weight = next(self.parameters())
        shape = (batch_size, self.num_layers, self.num_nodes, self.size)
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(shape), weight.new_zeros(shape))
        else:
            return weight.new_zeros(shape)

    def _forward_tick(self, output, hidden):
        for ilay, layer in enumerate(self.layers):
            hidden[:, ilay] = layer(output, hidden[:, ilay])
            output = hidden[:, ilay]
        return output, hidden


class GraphRNNCell(nn.Module):
    def __init__(self, rnn_type, size, func, **func_args):
        assert rnn_type in ['RNN', 'GRU']
        super().__init__()
        self.rnn_type = rnn_type
        gate_size = size
        if self.rnn_type == 'RNN': self.activation = nn.Tanh()
        elif self.rnn_type == 'RNNReLU': self.activation = nn.ReLU()
        elif self.rnn_type == 'GRU': gate_size *= 3
        self.func_i = func(input_size=size, output_size=gate_size, **func_args)
        self.func_h = func(input_size=size, output_size=gate_size, **func_args)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input, hidden):
        if self.rnn_type in ['RNN', 'RNNReLU']:
            output = self.activation(self.func_i(input) + self.func_h(hidden))
        elif self.rnn_type is 'GRU':
            output = self._gru(input, hidden)
        output = self.layer_norm(output)
        return output

    def _gru(self, input, hidden):
        i_r, i_i, i_n = self.func_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.func_h(hidden).chunk(chunks=3, dim=-1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output
