import torch
import torch.nn as nn


class LinearDecoder(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        return self.linear(self.dropout(self.layer_norm(input)))


class GraphDecoder(nn.Module):
    def __init__(self, node_count, hidden_size, output_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm((node_count, hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        return self.linear(self.dropout(self.layer_norm(input)))


class Attn(nn.Module):
    def __init__(self, attn_type, input_size, output_size, dropout):
        pass


class RNNDecoder(nn.Module):
    def __init__(self, rnn, output_size):
        super().__init__()
        self.rnn = rnn
        self.linear = nn.Linear(self.rnn.hidden_size, output_size)
        self.dropout = nn.Dropout(self.rnn.dropout_prob)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.linear(self.dropout(output))
        return output, hidden


class RNNAttnDecoder(RNNDecoder):
    def __init__(self, rnn, attn_type, output_size):
        super().__init__(rnn, output_size)
        self.attention = Attention.GlobalAttention(
            attn_type, self.rnn.hidden_size)

    def forward(self, input, hidden, context):
        output, hidden = self.rnn(input, hidden)
        output, attn = self.attention(output, context)
        output = self.linear(self.dropout(output))
        return output, hidden, attn
