import torch
import torch.nn as nn

from models import Framework


class GraphRNN(nn.Module):
    def __init__(self, rnn_type, model_dim, num_layers, num_nodes,
                 func=nn.Linear, **func_args):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GraphRNNCell(rnn_type, model_dim, func, **func_args)
            for i in range(num_layers)
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
        gate_size = model_dim
        self.layer_norm = nn.LayerNorm(model_dim)
        if self.rnn_type == 'RNN': self.activation = nn.Tanh()
        elif self.rnn_type == 'RNNReLU': self.activation = nn.ReLU()
        elif self.rnn_type == 'GRU': gate_size *= 3
        self.func_i = self.build_func(input_size, gate_size)
        self.func_h = self.build_func(input_size, gate_size)

    def forward(self, input, hidden):
        if self.rnn_type in ['RNN', 'RNNReLU']:
            output = self.activation(self.func_i(input) + self.func_h(hidden))
        elif self.rnn_type is 'GRU':
            output = self.gru(input, hidden)
        output = self.layer_norm(output)
        return output

    def build_func(self, input_size, output_size):
        raise NotImplementedError('Implement function for RNN!')

    def gru(self, input, hidden):
        i_r, i_i, i_n = self.func_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.func_h(hidden).chunk(chunks=3, dim=-1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newhidden = torch.tanh(i_n + resetgate * h_n)
        output = newhidden + inputgate * (hidden - newhidden)
        return output
