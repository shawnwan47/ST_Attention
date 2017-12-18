import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import aeq
from UtilClass import *


class AttnLayer(nn.Module):
    def __init__(self, dim, adj=None, dropout=0.1):
        super(AttnLayer, self).__init__()
        self.attn = ContextAttention(dim, adj, dropout)
        self.feed_forward = PointwiseMLP(dim, adj, dropout)

    def forward(self, inp, mask=None):
        out, attn = self.attn(inp, mask)
        out = self.feed_forward(out)  # I guess I can remove this nonlinearity
        return out, attn


class MultiChannelContextAttention(nn.Module):
    def __init__(self, dim, channel=1, adj=None, dropout=0.1):
        super(MultiChannelContextAttention, self).__init__()
        self.dim = dim
        self.channel = channel
        self.attn_channel = nn.ModuleList([
            ContextAttention(dim, adj, dropout) for _ in range(channel)])
        self.attn_merge = ContextAttention(dim, adj, dropout)
        self.feed_forward = PointwiseMLP(dim, adj, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask=None):
        '''
        IN
        query: batch x length_query x dim
        context: batch x channel x length_context x dim
        mask: length_query x length_context
        mask_context: length_context x length_context
        MID
        out_channel: batch*length_query x channel x dim
        OUT
        out: batch x length_query x dim
        attn_merge: batch x length_query x channel
        attn_channel: batch x channel x length_query x length_context
        '''
        batch, length_query, dim = query.size()
        batch_, channel, length_context, dim_ = context.size()
        aeq(batch, batch_)
        aeq(channel, self.channel)
        aeq(dim, dim_)

        out_channel, attn_channel = [], []
        for i in range(self.channel):
            out, attn = self.attn_channel[i](
                query, context[:, i].contiguous(), mask)
            out_channel.append(out)
            attn_channel.append(attn)
        out_channel = torch.stack(out_channel, -2).view(-1, channel, dim)
        query = query.contiguous().view(-1, 1, dim)
        out, attn_merge = self.attn_merge(query.view(-1, 1, dim), out_channel)
        attn_channel = torch.stack(attn_channel, 1)
        # out = self.feed_forward(out)  # guess I could remove nonlinearity
        out = out.view(batch, length_query, dim)
        return out, attn_merge, attn_channel


class ContextAttention(nn.Module):
    def __init__(self, dim, adj=None, dropout=0.1):
        super(ContextAttention, self).__init__()
        self.dim = dim
        self.w_k = BottleSparseLinear(dim, dim, adj, bias=False)
        self.w_v = BottleSparseLinear(dim, dim, adj, bias=False)
        self.layer_norm = BottleLayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.sm = nn.Softmax(2)
        self.feed_forward = PointwiseMLP(dim, adj, dropout)

    def forward(self, query, context, mask=None):
        '''
        query: batch x len x dim
        context: batch x len_ x dim
        '''
        key, val = self.w_k(context), self.w_v(context)
        score = torch.bmm(query, key.transpose(1, 2))
        score /= math.sqrt(self.dim)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(score)
        out = torch.bmm(attn, val)
        # out = self.dropout(out)
        out = self.layer_norm(out + query)
        out = self.feed_forward(out)  # really?
        return out, attn


class MultiChannelSelfAttention(nn.Module):
    def __init__(self, dim, channel=1, adj=None, dropout=0.1):
        super(MultiChannelSelfAttention, self).__init__()
        self.dim = dim
        self.channel = channel
        self.attn = nn.ModuleList([
            SelfAttention(dim, adj, dropout) for _ in range(channel)])

    def forward(self, inp, mask=None):
        '''
        inp: batch x channel x length x dim
        out: batch x channel x length x dim
        attn: batch x channel x length x length
        '''
        outs, attns = [], []
        for i in range(self.channel):
            out, attn = self.attn[i](inp[:, i].contiguous(), mask)
            outs.append(out)
            attns.append(attn)
        out = torch.stack(outs, 1)
        attn = torch.stack(attns, 1)
        return out, attn


class SelfAttention(nn.Module):
    def __init__(self, dim, adj=None, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.w_1 = BottleSparseLinear(self.dim, self.dim, adj=adj)
        self.w_2 = BottleLinear(self.dim, 1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.sm = nn.Softmax(2)

    def forward(self, inp, mask=None):
        '''
        inp: batch x length x dim
        mask: length x length
        score: batch x length
        attn: batch x length x length
        out: batch x length x dim
        '''
        batch, length, dim = inp.size()
        aeq(dim, self.dim)
        score = self.activation(self.w_1(inp))
        score = self.w_2(self.dropout(score))
        attn = score.repeat(1, 1, length).transpose(1, 2)
        if mask is not None:
            aeq(length, mask.size(0), mask.size(1))
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        out = torch.bmm(attn, inp).view(batch, length, dim)
        return out, attn


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


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, head=1, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadedAttention, must be divisible by head.
        '''
        assert dim % head == 0, '{}, {}'.format(dim, head)
        super(MultiHeadedAttention, self).__init__()
        self.dim = dim
        self.head = head
        self.dim_head = dim // head
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_v = BottleLinear(dim, dim, bias=False)
        self.w_q = BottleLinear(dim, dim, bias=False)
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
        key_up = shape_projection(self.w_k(inp))
        value_up = shape_projection(self.w_v(inp))
        query_up = shape_projection(self.w_q(inp))

        scaled = torch.bmm(query_up, key_up.transpose(1, 2))
        scaled = scaled / math.sqrt(self.dim_head)
        bh, l, dim_head = scaled.size()
        b = bh // self.head
        if mask is not None:
            scaled = scaled.view(b, self.head, l, dim_head)
            # mask = mask.expand_as(scaled)
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


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ['dot', 'general', 'mlp']

        if self.attn_type == 'general':
            self.w_in = BottleLinear(dim, dim, bias=False)
        elif self.attn_type == 'mlp':
            self.w_c = BottleLinear(dim, dim, bias=False)
            self.w_q = BottleLinear(dim, dim, bias=True)
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
                hid_ = self.w_in(inp).transpose(1, 2)
                return torch.bmm(inp, hid_)
            else:
                hid_ = inp.transpose(1, 2).contiguous()
                return torch.bmm(inp, hid_)
        else:
            dim = self.dim
            wq = self.w_q(inp).view(batch, length, 1, dim)
            uh = self.w_c(inp).view(batch, 1, length, dim)
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

