import torch
import torch.nn


class GCRNNCell(nn.Module):
    def __init__(self, rnn_type, input_size, output_size,
                 func, *args, **kwargs):
        assert rnn_type in ['RNNtanh', 'RNNrelu', 'GRU', 'LSTM']
        super().init()
        self.rnn_type = rnn_type
        gate_size = output_size
        if self.rnn_type == 'GRU': gate_size *= 3
        elif self.rnn_type == 'LSTM': gate_size *= 4
        self.func_i = func(input_size, gate_size, *args, **kwargs)
        self.func_h = func(output_size, gate_size, *args, **kwargs)

    def forward(self, input, hidden):
        if self.rnn_type == 'RNNtanh':
            output = F.tanh(self.rnn(input, hidden))
        elif self.rnn_type == 'RNNrelu':
            output = F.relu(self.rnn(input, hidden))
        elif self.rnn_type == 'GRU':
            output = self.gru(input, hidden)
        else:
            output = self.lstm(input, hidden)
        return output

    def rnn(self, input, hidden):
        return self.func_i(input) + self.func_h(hidden)

    def gru(self, input, hidden):
        gi = self.func_i(input)
        gh = self.func_h(hidden)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output

    def lstm(self, input, hidden):
        hx, cx = hidden
        gates = self.func_i(input) + self.func_h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy


class GCRNN(nn.Module):
    def __init__(self, rnn_type, node_count,
                 input_size, hidden_size, num_layers, p_dropout,
                 func, *args, **kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        self.node_count = node_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            GCRNNCell(rnn_type, input_size, hidden_size, func, *args, **kwargs)
        )
        self.layers.extend((
            GCRNNCell(rnn_type, hidden_size, hidden_size, func, *args, **kwargs)
            for i in range(num_layers - 1)
        ))
        self.dropout = nn.Dropout(p_dropout)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        size = (batch_size, self.num_layers, self.node_count, self.hidden_size)
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(size), weight.new_zeros(size))
        else:
            return weight.new_zeros(size)

    def forward(self, input, hidden=None):
        batch_size, seq_len, node_count, input_size = input.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        output = []
        for idx in range(seq_len):
            output_i = input[idx]
            for ilay, layer in enumerate(self.layers):
                if self.rnn_type == 'LSTM':
                    hidden[0][:, ilay], hidden[1][:, ilay] = layer(
                        output_i, (hidden[0][:, ilay], hidden[1][:, ilay]))
                    output_i = hidden[0][:, ilay]
                else:
                    hidden[:, ilay] = layer(output_i, hidden[:, ilay])
                    output_i = hidden[:, ilay]
                if ilay < self.num_layers - 1:
                    output_i = self.dropout(output_i)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden
