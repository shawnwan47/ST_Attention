import torch
import torch.nn as nn


class LNLinearReg(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        return self.linear(self.layer_norm(self.dropout(input)))


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
