import torch
import torch.nn as nn

from models import Attention


class GraphAttention(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout=0.1, mask=None):
        super().__init__()
        self.attention = Attention.MultiAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout,
            output_size=output_size
        )
        self.linear_query = nn.Linear(input_size, output_size, bias=False)
        self.register_buffer('mask', mask)

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x input_size
        '''
        context, attn = self.attention(input, input, input, self.mask)
        output = self.linear_query(input) + context
        return output, attn


class GraphRelativeAttention(GraphAttention):
    def __init__(self, input_size, output_size, head_count, dropout, adj, mask=None):
        super().__init__(input_size, output_size, head_count, dropout, mask)
        self.attention = Attention.MultiRelativeAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout,
            adj=adj,
            output_size=output_size
        )


class GARNNCell(nn.Module):
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
        if self.rnn_type == 'RNN':
            output_i, attn_i = self.gc_i(input)
            output_h, attn_h = self.gc_h(hidden)
            output = torch.tanh(output_i + output_h)
        elif self.rnn_type == 'GRU':
            output = self._gru(input, hidden)
        output = self.layer_norm(output)
        return output

    def _gru(self, input, hidden):
        output_i, attn_i = self.gc_i(input)
        output_h, attn_h = self.gc_h(input)
        i_r, i_i, i_n = output_i.chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = output_h.chunk(chunks=3, dim=-1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        output = newgate + inputgate * (hidden - newgate)
        return output


class GARNN(DCRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.layers = nn.ModuleList([
            GARNNCell(rnn_type, size, gc_func, gc_kwargs)
            for i in range(num_layers)
        ])


class GARNNDecoder(GARNN):
    def __init__(self, rnn_type, num_nodes, size, out_size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.gat = gc_func(input_size=size, output_size=1, **gc_kwargs)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        output, attn = self.gat(output)
        return output, hidden, attn
