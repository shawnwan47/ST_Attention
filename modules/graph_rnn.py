import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphRNN(nn.Module):
    def __init__(self, rnn_type, model_dim, num_layers, num_nodes, func, **kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GraphRNNCell(rnn_type, model_dim) for _ in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        _, seq_len = input.size(0), input.size(1)
        if hidden is None:
            hidden = self.init_hidden()
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

    def init_hidden(self):
        weight = next(self.parameters())
        shape = (1, self.num_layers, self.num_nodes, self.model_dim)
        if self.rnn_type == 'LSTM':
            return weight.new_zeros(2, *shape)
        else:
            return weight.new_zeros(shape)


class GraphRNNCell(nn.Module):
    def __init__(self, rnn_type, model_dim, func, **func_args):
        assert rnn_type in ['RNN', 'GRU', 'LSTM']
        super().__init__()
        self.rnn_type = rnn_type
        self.layer_norm = nn.LayerNorm(model_dim)
        if self.rnn_type == 'RNN':
            out_dim = model_dim
        elif self.rnn_type == 'GRU':
            out_dim = model_dim * 3
        else:
            out_dim = model_dim * 4
        self.func_i = func(model_dim, out_dim, **func_args)
        self.func_h = func(model_dim, out_dim, **func_args)

    def forward(self, input, hidden):
        if self.rnn_type is 'RNN':
            output = F.tanh(self.func_i(input) + self.func_h(hidden))
        elif self.rnn_type is 'GRU':
            output = self.gru(input, hidden)
        else:
            output = self.lstm(input, hidden)
        output = self.layer_norm(output)
        return output

    def gru(self, input, hidden):
        i_r, i_i, i_n = self.func_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.func_h(hidden).chunk(chunks=3, dim=-1)
        gate_r = F.sigmoid(i_r + h_r)
        gate_i = F.sigmoid(i_i + h_i)
        i_new = F.tanh(i_n + gate_r * h_n)
        output = i_new + gate_i * (hidden - i_new)
        return output

    def lstm(self, input, hidden):
        hidden, cell = hidden
        i_i, i_o, i_f, i_n = self.func_i(input).chunk(chunks=4, dim=-1)
        h_i, h_o, h_f, h_n = self.func_h(hidden).chunk(chunks=4, dim=-1)
        gate_i = F.sigmoid(i_i + h_i)
        gate_o = F.sigmoid(i_o + h_o)
        gate_f = F.sigmoid(i_f + h_f)
        cell_new = F.tanh(i_n + h_n)
        cell = gate_i * cell_new + gate_f * cell
        hidden = gate_o * F.tanh(cell)
        return hidden, cell
