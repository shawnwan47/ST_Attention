import torch
import torch.nn as nn
import torch.nn.functional as F


class GCRNNCell(nn.Module):
    def __init__(self, rnn_type, size, gc_func, gc_kwargs):
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

    def forward(self, input, hidden):
        if self.rnn_type in ['RNN', 'RNNReLU']:
            output = self.activation(self.gc_i(input) + self.gc_h(hidden))
        elif self.rnn_type == 'GRU':
            output = self._gru(input, hidden)
        output = self.layer_norm(output)
        return output

    def _gru(self, input, hidden):
        i_r, i_i, i_n = self.gc_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.gc_h(hidden).chunk(chunks=3, dim=-1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output


class GARNNCell(GCRNNCell):
    def forward(self, input, hidden):
        if self.rnn_type == 'RNN':
            output_i, attn_i = self.gc_i(input)
            output_h, attn_h = self.gc_h(hidden)
            output = F.tanh(output_i + output_h)
        elif self.rnn_type == 'GRU':
            output = self._gru(input, hidden)
        output = self.layer_norm(output)
        return output

    def _gru(self, input, hidden):
        output_i, attn_i = self.gc_i(input)
        output_h, attn_h = self.gc_h(input)
        i_r, i_i, i_n = output_i.chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = output_h.chunk(chunks=3, dim=-1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output
