import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import Layers

from Attention import GlobalAttention
from Utils import aeq


class EncoderBase(nn.Module):
    """
    EncoderBase class for sharing code among various encoder.
    """

    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        raise NotImplementedError


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """

    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        assert args.rnn_type in ['LSTM', 'GRU']
        self.rnn_type = args.rnn_type
        self.ndim = args.ndim
        self.nhid = args.nhid
        self.nlay = args.nlay
        self.rnn = getattr(nn, self.rnn_type)(
            args.ndim, args.nhid, args.nlay,
            dropout=args.pdrop, bidirectional=args.bidirectional)

    def forward(self, inputs, hiddens):
        return self.rnn(inputs, hiddens)

    def encode(self, inputs, hiddens):
        context, hiddens = self.rnn(inputs, hiddens)
        return context.transpose(0, 1).contiguous(), hiddens

    def initHidden(self, inputs):
        hiddens = Variable(torch.zeros(self.nlay, inputs.size(1), self.nhid))
        return hiddens.cuda() if inputs.is_cuda else hiddens


class RNNAttn(RNN):
    def __init__(self, args):
        super(RNNAttn, self).__init__(args)
        self.attention_type = args.attention_type
        self.attention_model = GlobalAttention(args.nhid, self.attention_type)
        self.context_length = args.context_length

    def forward(self, inputs, hiddens, context):
        context_, hiddens = self.encode(inputs, hiddens)
        bsz, srcL, ndim = context.size()
        bsz_, tgtL, ndim_ = context_.size()
        aeq(bsz, bsz_)
        aeq(ndim, ndim_)
        context = torch.cat((context, context_), 1)
        # mask unseen context
        mask = torch.zeros(bsz, tgtL, srcL + tgtL).type(torch.ByteTensor)
        for i in range(tgtL):
            mask[:, i, srcL + i:] = 1
        # mask long-term context
        if self.context_length:
            for i in range(tgtL):
                end = srcL + i - self.context_length
                if end > 0:
                    mask[:, i, :end] = 1
        mask = mask.cuda() if context.is_cuda else mask
        self.attention_model.applyMask(mask)
        # compute outputs and attentions
        outputs, attentions = self.attention_model(context_, context)
        return outputs, hiddens, context, attentions


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.attention = args.attention
        if self.attention:
            self.rnn = RNNAttn(args)
        else:
            self.rnn = RNN(args)
        self.dropout = nn.Dropout(args.pdrop)
        self.linear_out = nn.Linear(args.nhid, args.ndim)

    def forward(self, src, tgt, teach=True):
        srcL, bsz, ndim = src.size()
        tgtL, bsz_, ndim_ = tgt.size()
        aeq(bsz, bsz_)
        aeq(ndim, ndim_)
        hiddens = self.rnn.initHidden(src)
        context, hiddens = self.rnn.encode(src, hiddens)
        if teach:
            if self.attention:
                outputs, hiddens, context, attentions = self.rnn(
                    tgt, hiddens, context)
            else:
                outputs, hiddens = self.rnn(tgt, hiddens)
            outputs = tgt + self.linear_out(self.dropout(outputs))
        else:
            # inputfeeding prediction
            outputs = tgt.clone()
            attentions = Variable(torch.zeros(tgtL, bsz, srcL + tgtL))
            inp = tgt[0].unsqueeze(0)
            for i in range(tgtL):
                if self.attention:
                    res, hiddens, context, attn = self.rnn(
                        inp, hiddens, context)
                    attentions[i, :, :srcL + i] = attn[0, :, :-1]
                else:
                    res, hiddens = self.rnn(inp, hiddens)
                outputs[i] = inp + self.linear_out(self.dropout(res))
                inp = outputs[i].clone().unsqueeze(0)
        if self.attention:
            return outputs, hiddens, context, attentions
        else:
            return outputs, hiddens


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.nlay = args.nlay
        self.gc_in = Layers.GraphConvolution(args.past, args.nhid)
        self.gc_hid = Layers.GraphConvolution(args.nhid, args.nhid)
        self.gc_out = Layers.GraphConvolution(args.nhid, args.future)
        self.activation = nn.Sequential(nn.ReLU(), nn.Dropout(args.pdrop))

    def forward(self, x, adj):
        x = self.activation(self.gc_in(x, adj))
        for l in range(self.nlay - 2):
            x = self.activation(self.gc_hid(x, adj))
        return self.gc_out(x, adj)


class GAT(nn.Module):
    '''
    A simplified 2-layer GAT
    '''

    def __init__(self, ninp, nhid, att_heads, nout):
        super(GAT, self).__init__()
        self.ninp = ninp
        self.nout = nout
        self.nhid = nhid
        self.att_heads = att_heads

        self.gat1 = Layers.GraphAttention(ninp, nhid, att_heads, True)
        self.gat2 = Layers.GraphAttention(self.gat1.nout, nout)

    def forward(self, inp, adj):
        out, att1 = self.gat1(inp, adj)
        out, att2 = self.gat2(out, adj)
        return out, att1, att2
