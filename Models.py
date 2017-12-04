import torch
import torch.nn as nn
from torch.autograd import Variable

import Layers

from Attention import GlobalAttention
from Utils import aeq
import UtilClass


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


class Transformer(nn.Module):
    """
    The Transformer decoder for Spatial-Temporal Attention Model
    """

    def __init__(self, ndim, nhid, nlay, pdrop):
        super(Transformer, self).__init__()
        self.linear_in = UtilClass.BottleLinear(ndim, nhid)
        self.linear_out = UtilClass.BottleLinear(nhid, ndim)
        self.transformer_layers = nn.ModuleList(
            [Layers.TransformerLayer(nhid, pdrop)
             for _ in range(nlay)])

    def encode(self, inputs):
        outputs = self.linear_in(inputs)
        for layer in self.transformer_layers:
            outputs, attn = layer.encode(outputs)
        return outputs, attn

    def forward(self, inputs, context):
        # CHECKS
        aeq(inputs.dim(), 3)
        input_len, input_batch, _ = inputs.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END CHECKS
        outputs = self.linear_in(inputs)

        outputs = outputs.transpose(0, 1).contiguous()
        context = context.transpose(0, 1).contiguous()

        for layer in self.transformer_layers:
            outputs, attn = layer(outputs, context)

        # Process the result and update the attentions.
        outputs = outputs.transpose(0, 1).contiguous()
        outputs = inputs + self.linear_out(outputs)
        return outputs, attn


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
