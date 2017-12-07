import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import UtilClass
<<<<<<< HEAD
from MultiHeadedAttn import MultiHeadedAttention
from Utils import aeq
from Consts import MAX_SEQ_LEN


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network."""

    def __init__(self, size, hidden_size=None, dropout=0.1):
        """
        Args:
            size(int): the size of inputs for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        """
        super(PositionwiseFeedForward, self).__init__()
        hidden_size = size if hidden_size is None else hidden_size
        self.w_1 = UtilClass.BottleLinear(size, hidden_size)
        self.w_2 = UtilClass.BottleLinear(hidden_size, size)
        self.layer_norm = UtilClass.BottleLayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        outputs = self.dropout(self.w_2(self.relu(self.w_1(inputs))))
        return self.layer_norm(outputs + inputs)


class TransformerLayer(nn.Module):
    def __init__(self, dim=1024, dropout=0.1, head_count=8):
        """
        layer for Transformer in Models.py
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the inputs size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerLayer, self).__init__()
        self.attn = MultiHeadedAttention(head_count, dim, p=dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dropout=dropout)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        mask = self._get_attn_subsequent_mask(MAX_SEQ_LEN)
        self.register_buffer('mask', mask)

    def encode(self, inputs, mask=None):
        outputs, attn = self.attn(inputs, inputs, inputs, mask)
        outputs = self.feed_forward(outputs)
        return outputs, attn

    def forward(self, inputs, context, mask_src=None, mask_tgt=None):
        # Args Checks
        input_batch, input_len, _ = inputs.size()
        contxt_batch, contxt_len, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Checks

        mask_dec = torch.gt(self.mask[:, :input_len, :input_len], 0)
        query, attn = self.encode(inputs, mask_dec)
        mid, attn = self.attn(context, context, query, mask_src)
        outputs = self.feed_forward(mid)

        return outputs, attn
=======
>>>>>>> 4040e05dbfdeb87d79b41c8070ec3291c5e46673


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, nin, nout, bias=True):
        super(GraphConvolution, self).__init__()
        self.nin = nin
        self.nout = nout
        self.linear = UtilClass.BottleLinear(nin, nout)

    def forward(self, inputs, adj):
        '''
        inputs: batch x node x feature
        adj: node x node
        '''
        pass

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.nin) + ' -> ' \
            + str(self.nout) + ')'


class GraphAttention(nn.Module):
<<<<<<< HEAD
    def __init__(self, nin, nout, nhead, dropout):
        super(GraphAttention, self).__init__()
        self.nin = nin
        self.nout = nout
        self.linear = UtilClass.BottleLinear(nin, nout)
        self.nhead = nhead
        self.self_attn = MultiHeadedAttention(nhead, nout)

        self.nout = self.nout * self.nhead
        self.activation = nn.Sequential(nn.Dropout(dropout), nn.ReLU())
=======
    def __init__(self, ninp, nfeat, nheads=1, nonlinear=False):
        super(GraphAttention, self).__init__()
        self.ninp = ninp
        self.nfeat = nfeat
        self.nheads = nheads

        self.kernels = []
        self.att_kernels = []

        for head in range(nheads):
            self.kernels.append(nn.Linear(ninp, nfeat))
            self.att_kernels.append(nn.Linear(2 * nfeat, 1))

        self.nout = self.nfeat * self.nheads
        self.activation = nn.ELU() if nonlinear else None
>>>>>>> 4040e05dbfdeb87d79b41c8070ec3291c5e46673

    def forward(self, inp, adj):
        '''
        inp: bsz x nnode x nin
        adj: nnode x nnode
        out: bsz x nnode x nout
        '''
        bsz, nnode, nin = inp.size()
        assert adj.size(0) == nnode
        outs, atts = [], []
<<<<<<< HEAD
        for head in range(self.nhead):
=======
        for head in range(self.nheads):
>>>>>>> 4040e05dbfdeb87d79b41c8070ec3291c5e46673
            kernel = self.kernels[head]
            att_kernel = self.att_kernels[head]
            out = kernel(inp)

            h = [out for _ in range(nnode)]
            h_i = torch.cat(h, -1).view(bsz, nnode ** 2, -1)
            h_j = torch.cat(h, 1)
            att = att_kernel(torch.cat((h_i, h_j), -1)).view(bsz, nnode, -1)
            att = F.softmax(att[:, adj])
            outs.append(torch.bmm(att, out))
            atts.append(att)
        outs = torch.cat(outs, -1)
        if self.activation:
            outs = self.activation(outs)
        return outs, atts
