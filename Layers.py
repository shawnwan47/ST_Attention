import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import aeq, _get_attn_subsequent_mask
from UtilClass import BottleLinear, BottleLayerNorm, BottleSoftmax


class RNNBase(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        super(RNNBase, self).__init__()
        assert rnn_type in ['LSTM', 'GRU', 'RNN']
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = getattr(nn, self.rnn_type)(
            input_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, input, hidden):
        return self.rnn(input, hidden)

    def encode(self, input, hidden):
        context, hidden = self.rnn(input, hidden)
        return context.transpose(0, 1).contiguous(), hidden

    def initHidden(self, input):
        batch = input.size(1)
        h_0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
        return h_0.cuda() if input.is_cuda else h_0


class RNNAttn(RNNBase):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout,
                 attn_type):
        super(RNNAttn, self).__init__(
            rnn_type, input_size, hidden_size, num_layers, dropout)
        self.attn_type = attn_type
        self.attn_model = GlobalAttention(hidden_size, attn_type)
        mask = _get_attn_subsequent_mask()
        self.register_buffer('mask', mask)

    def forward(self, input, hidden, context):
        context_, hidden = self.encode(input, hidden)
        bsz, srcL, input_size = context.size()
        bsz_, tgtL, ndim_ = context_.size()
        aeq(bsz, bsz_)
        aeq(input_size, ndim_)
        context = torch.cat((context, context_), 1)
        # mask subsequent context
        mask = self.mask[0, srcL:srcL + tgtL, :srcL + tgtL]
        self.attn_model.applyMask(mask)
        # compute outputs and attns
        outputs, attns = self.attn_model(context_, context)
        return outputs, hidden, context, attns


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network."""

    def __init__(self, size, hidden_size=None, dropout=0.1):
        """
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        """
        super(PositionwiseFeedForward, self).__init__()
        hidden_size = size if hidden_size is None else hidden_size
        self.w_1 = BottleLinear(size, hidden_size)
        self.w_2 = BottleLinear(hidden_size, size)
        self.layer_norm = BottleLayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input):
        outputs = self.dropout(self.w_2(self.relu(self.w_1(input))))
        return self.layer_norm(outputs + input)


class TransformerLayer(nn.Module):
    def __init__(self, dim=1024, dropout=0.1, head_count=8):
        super(TransformerLayer, self).__init__()
        self.attn = MultiHeadedAttention(head_count, dim, p=dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dropout=dropout)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        mask = _get_attn_subsequent_mask()
        self.register_buffer('mask', mask)

    def encode(self, input, mask=None):
        outputs, attn = self.attn(input, input, input, mask)
        outputs = self.feed_forward(outputs)
        return outputs, attn

    def forward(self, input, context, mask_src=None, mask_tgt=None):
        # Args Checks
        input_batch, input_len, _ = input.size()
        contxt_batch, contxt_len, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Checks

        mask_dec = torch.gt(self.mask[:, :input_len, :input_len], 0)
        query, attn = self.encode(input, mask_dec)
        mid, attn = self.attn(context, context, query, mask_src)
        outputs = self.feed_forward(mid)

        return outputs, attn


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type="dot"):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ["dot", "general", "mlp"]

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = BottleLinear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(2)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim
        h_s (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x tgt_len x src_len:
            raw attention scores for each src index
        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, context, coverage=None):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output.
        context (FloatTensor): batch x src_len x dim: src hidden states
        coverage (FloatTensor): None (not supported yet)
        """
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)

        if self.mask is not None:
            mask = self.mask.expand_as(align)
            align.data.masked_fill_(mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch * targetL, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, align_vectors


class MultiHeadedAttention(nn.Module):
    '''
    "Attention is All You Need".
    '''

    def __init__(self, head_count, model_dim, p=0.1):
        """
        Args:
            head_count(int): number of parallel heads.
            model_dim(int): the dimension of keys/values/queries in this
                MultiHeadedAttention, must be divisible by head_count.
        """
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = BottleLinear(model_dim,
                                        head_count * self.dim_per_head,
                                        bias=False)
        self.linear_values = BottleLinear(model_dim,
                                          head_count * self.dim_per_head,
                                          bias=False)
        self.linear_query = BottleLinear(model_dim,
                                         head_count * self.dim_per_head,
                                         bias=False)
        self.sm = nn.Softmax(2)
        self.activation = nn.ReLU()
        self.layer_norm = BottleLayerNorm(model_dim)
        self.dropout = nn.Dropout(p)
        self.res_dropout = nn.Dropout(p)

    def forward(self, key, value, query, mask=None):
        # CHECKS
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        aeq(batch, batch_)
        aeq(k_len, k_len_)
        aeq(d, d_)
        batch_, q_len, d_ = query.size()
        aeq(batch, batch_)
        aeq(d, d_)
        aeq(self.model_dim % 8, 0)
        if mask is not None:
            batch_, q_len_, k_len_ = mask.size()
            aeq(batch_, batch)
            aeq(k_len_, k_len)
            aeq(q_len_ == q_len)
        # END CHECKS

        def shape_projection(x):
            b, l, d = x.size()
            return x.view(b, l, self.head_count, self.dim_per_head) \
                .transpose(1, 2).contiguous() \
                .view(b * self.head_count, l, self.dim_per_head)

        def unshape_projection(x, q):
            b, l, d = q.size()
            return x.view(b, self.head_count, l, self.dim_per_head) \
                    .transpose(1, 2).contiguous() \
                    .view(b, l, self.head_count * self.dim_per_head)

        residual = query
        key_up = shape_projection(self.linear_keys(key))
        value_up = shape_projection(self.linear_values(value))
        query_up = shape_projection(self.linear_query(query))

        scaled = torch.bmm(query_up, key_up.transpose(1, 2))
        scaled = scaled / math.sqrt(self.dim_per_head)
        bh, l, dim_per_head = scaled.size()
        b = bh // self.head_count
        if mask is not None:
            scaled = scaled.view(b, self.head_count, l, dim_per_head)
            mask = mask.unsqueeze(1).expand_as(scaled)
            scaled = scaled.masked_fill(mask, -float('inf')) \
                           .view(bh, l, dim_per_head)
        attn = self.sm(scaled)

        drop_attn = self.dropout(attn)
        # values : (batch * 8) x qlen x dim
        out = unshape_projection(torch.bmm(drop_attn, value_up), residual)
        drop_out = self.dropout(out)
        # Residual and layer norm
        ret = self.layer_norm(drop_out + residual)

        # CHECK
        batch_, q_len_, d_ = ret.size()
        aeq(q_len, q_len_)
        aeq(batch, batch_)
        aeq(d, d_)
        # END CHECK
        return ret, attn
