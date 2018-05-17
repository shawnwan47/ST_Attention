import torch
import torch.nn as nn
from torch.nn import functional as F


class GCRNNCell(nn.Module):
    def __init__(self, rnn_type, input_size, output_size,
                 func, **kwargs):
        assert rnn_type in ['RNN', 'RNNReLU', 'GRU', 'LSTM']
        super().__init__()
        self.rnn_type = rnn_type
        gate_size = output_size
        if self.rnn_type == 'GRU': gate_size *= 3
        elif self.rnn_type == 'LSTM': gate_size *= 4
        self.gc_i = func(input_size, gate_size, **kwargs)
        self.gc_h = func(output_size, gate_size, **kwargs)

    def forward(self, input, hidden):
        if self.rnn_type == 'RNN':
            output = F.tanh(self.rnn(input, hidden))
        elif self.rnn_type == 'RNNReLU':
            output = F.relu(self.rnn(input, hidden))
        elif self.rnn_type == 'GRU':
            output = self.gru(input, hidden)
        else:
            output = self.lstm(input, hidden)
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
        ingate, forgetgate, cellgate, outgate = gates.chunk(chunks=4, dim=-1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy


class GCRNN(nn.Module):
    def __init__(self, rnn_type, node_count,
                 input_size, hidden_size, num_layers, dropout,
                 func, **kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        self.node_count = node_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            GCRNNCell(rnn_type, input_size, hidden_size, func, **kwargs)
        )
        self.layers.extend((
            GCRNNCell(rnn_type, hidden_size, hidden_size, func, **kwargs)
            for i in range(num_layers - 1)
        ))
        self.dropout = nn.Dropout(dropout)
        self.dropout_prob = dropout

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        shape = (batch_size, self.num_layers, self.node_count, self.hidden_size)
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(shape), weight.new_zeros(shape))
        else:
            return weight.new_zeros(shape)

    def forward(self, input, hidden=None):
        batch_size, seq_len, node_count, input_size = input.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        output = []
        for idx in range(seq_len):
            output_i, hidden = self.forward_layers(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden

    def forward_layers(self, output, hidden):
        for ilay, layer in enumerate(self.layers):
            if self.rnn_type == 'LSTM':
                hidden[0][:, ilay], hidden[1][:, ilay] = layer(
                    output, (hidden[0][:, ilay], hidden[1][:, ilay]))
                output = hidden[0][:, ilay]
            else:
                hidden[:, ilay] = layer(output, hidden[:, ilay])
                output = hidden[:, ilay]
            if ilay < self.num_layers - 1:
                output = self.dropout(output)
        return output, hidden
