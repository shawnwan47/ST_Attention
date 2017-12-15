import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import aeq
from UtilClass import BottleLinear, BottleLayerNorm


class PositionwiseFeedForward(nn.Module):
    ''' A two-layer Feed-Forward-Network.'''

    def __init__(self, size, hidden_size=None, dropout=0.1):
        '''
        Args:
            size(int): the size of inp for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        '''
        super(PositionwiseFeedForward, self).__init__()
        hidden_size = size if hidden_size is None else hidden_size
        self.w_1 = BottleLinear(size, hidden_size)
        self.w_2 = BottleLinear(hidden_size, size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(size)

    def forward(self, inp):
        out = self.dropout(self.w_2(self.relu(self.w_1(inp))))
        return self.layer_norm(out + inp)


class RNNBase(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        super(RNNBase, self).__init__()
        assert rnn_type in ['RNN', 'GRU', 'LSTM']
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = getattr(nn, self.rnn_type)(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True)

    def forward(self, inp, hid):
        out, hid = self.rnn(inp, hid)
        return out, hid

    def initHidden(self, inp):
        h = Variable(torch.zeros(
            self.num_layers, inp.size(0), self.hidden_size)).cuda()
        if self.rnn_type == 'LSTM':
            return (h, h.clone())
        else:
            return h


class RNNAttn(RNNBase):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout,
                 attn_type):
        super(RNNAttn, self).__init__(
            rnn_type, input_size, hidden_size, num_layers, dropout)
        self.attn_type = attn_type
        self.attn = GlobalAttention(hidden_size, attn_type)

    def forward(self, inp, hid, mask):
        out, hid = self.rnn(inp, hid)
        out, attn = self.attn(out, mask)
        return out, hid, attn


class AttentionLayer(nn.Module):
    def __init__(self, dim, head=8, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadedAttention(head, dim, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dropout=dropout)

    def forward(self, inp, mask=None):
        hid, attn = self.attn(inp, mask)
        out = self.feed_forward(hid)
        return out, attn


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ['dot', 'general', 'mlp']

        if self.attn_type == 'general':
            self.linear_in = BottleLinear(dim, dim, bias=False)
        elif self.attn_type == 'mlp':
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = BottleLinear(dim, dim, bias=True)
            self.v = BottleLinear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == 'mlp'
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(2)
        self.tanh = nn.Tanh()

    def score(self, inp):
        '''
        inp (FloatTensor): batch x length x dim
        returns scores (FloatTensor): batch x length x length:
        '''
        batch, length, dim = inp.size()

        if self.attn_type in ['general', 'dot']:
            if self.attn_type == 'general':
                hid_ = self.linear_in(inp).transpose(1, 2)
                return torch.bmm(inp, hid_)
            else:
                hid_ = inp.transpose(1, 2).contiguous()
                return torch.bmm(inp, hid_)
        else:
            dim = self.dim
            wq = self.linear_query(inp).view(batch, length, 1, dim)
            uh = self.linear_context(inp).view(batch, 1, length, dim)
            wq = wq.expand(batch, length, length, dim)
            uh = uh.expand(batch, length, length, dim)
            wquh = self.tanh(wq + uh).view(-1, dim)
            return self.v(wquh).view(batch, length, length)

    def forward(self, inp, mask=None):
        '''
        inp (FloatTensor): batch x length x dim: decoder's rnn's output.
        '''
        batch, length, dim = inp.size()

        align = self.score(inp.contiguous())
        if mask is not None:
            mask = self.mask.expand_as(align)
            align.data.masked_fill_(mask, -float('inf'))

        attn = self.sm(align)
        c = torch.bmm(attn, inp)
        concat_c = torch.cat([c, inp], 2).view(batch * length, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, length, dim)
        if self.attn_type in ['general', 'dot']:
            attn_h = self.tanh(attn_h)

        return attn_h, attn


class MultiHeadedAttention(nn.Module):
    '''
    'Attention is All You Need'.
    '''

    def __init__(self, head, dim, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadedAttention, must be divisible by head.
        '''
        assert dim % head == 0, '{}, {}'.format(dim, head)
        self.dim_head = dim // head
        self.dim = dim

        super(MultiHeadedAttention, self).__init__()
        self.head = head

        self.linear_keys = BottleLinear(dim, dim, bias=False)
        self.linear_values = BottleLinear(dim, dim, bias=False)
        self.linear_query = BottleLinear(dim, dim, bias=False)
        self.sm = nn.Softmax(2)
        self.activation = nn.ReLU()
        self.layer_norm = BottleLayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask=None):
        batch, length, dim = inp.size()

        def shape_projection(x):
            b, l, d = x.size()
            return x.view(b, l, self.head, self.dim_head) \
                .transpose(1, 2).contiguous() \
                .view(b * self.head, l, self.dim_head)

        def unshape_projection(x, q):
            b, l, d = q.size()
            return x.view(b, self.head, l, self.dim_head) \
                    .transpose(1, 2).contiguous() \
                    .view(b, l, self.dim)

        residual = inp
        key_up = shape_projection(self.linear_keys(inp))
        value_up = shape_projection(self.linear_values(inp))
        query_up = shape_projection(self.linear_query(inp))

        scaled = torch.bmm(query_up, key_up.transpose(1, 2))
        scaled = scaled / math.sqrt(self.dim_head)
        bh, l, dim_head = scaled.size()
        b = bh // self.head
        if mask is not None:
            scaled = scaled.view(b, self.head, l, dim_head)
            mask = mask.unsqueeze(1).expand_as(scaled)
            scaled.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(scaled.view(bh, l, dim_head))

        # values : (batch * 8) x qlen x dim
        out = torch.bmm(self.dropout(attn), value_up)
        out = unshape_projection(out, residual)
        out = self.layer_norm(self.dropout(out) + residual)
        attn = attn.view(b, self.head, l, dim_head)

        # CHECK
        batch_, length_, dim_ = out.size()
        aeq(length, length_)
        aeq(batch, batch_)
        aeq(dim, dim_)
        # END CHECK
        return out, attn
