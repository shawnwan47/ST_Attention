import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphRNN(nn.Module):
    def __init__(self, rnn_type, rnn_cell, model_dim, num_layers, num_nodes):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            rnn_cell(rnn_type, model_dim) for _ in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size(0), input.size(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        output = []
        for idx in range(seq_len):
            output_i, hidden = self.forward_i(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden

    def forward_i(self, output, hidden):
        hidden_new = []
        for ilay, layer in enumerate(self.layers):
            output = layer(output, hidden[:, ilay])
            hidden_new.append(output)
            hidden_new = torch.stack(hidden_new, 1)
            return output, hidden_new

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        shape = (batch_size, self.num_layers, self.num_nodes, self.model_dim)
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(shape), weight.new_zeros(shape))
        else:
            return weight.new_zeros(shape)


class GraphRNNCell(nn.Module):
    def __init__(self, rnn_type, model_dim, func, **func_args):
        assert rnn_type in ['RNN', 'GRU', 'LSTM']
        super().__init__()
        self.rnn_type = rnn_type
        self.layer_norm = nn.LayerNorm(model_dim)
        if self.rnn_type == 'RNN':
            output_size = model_dim
        elif self.rnn_type == 'GRU':
            output_size = model_dim * 3
        else:
            output_size = model_dim * 4
        self.func_i = self.build_func(model_dim, output_size)
        self.func_h = self.build_func(model_dim, output_size)

    def forward(self, input, hidden):
        if self.rnn_type is 'RNN':
            output = F.tanh(self.func_i(input) + self.func_h(hidden))
        elif self.rnn_type is 'GRU':
            output = self.gru(input, hidden)
        else:
            output = self.lstm(input, hidden)
        output = self.layer_norm(output)
        return output

    def build_func(self, input_size, output_size):
        raise NotImplementedError('Implement function for GraphRNN!')

    def gru(self, input, hidden):
        input_r, input_i, input_n = self.func_i(input).chunk(chunks=3, dim=-1)
        hidden_r, hidden_i, hidden_n = self.func_h(hidden).chunk(chunks=3, dim=-1)
        gate_r = torch.sigmoid(input_r + hidden_r)
        gate_i = torch.sigmoid(input_i + hidden_i)
        input_new = torch.tanh(input_n + gate_r * hidden_n)
        output = input_new + gate_i * (hidden - input_new)
        return output

    def lstm(self, input, hidden):
        raise NotImplementedError('LSTM not implemented!')
