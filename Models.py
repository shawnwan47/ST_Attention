import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import Layers

from Attention import GlobalAttention


class RNNAttn(nn.Module):
    def __init__(self, ndim, nhid, nlay, pdrop=0.5, rnn_type='GRU',
                 attn=False, attn_type='general'):
        super(RNNAttn, self).__init__()
        assert rnn_type in ['LSTM', 'GRU']
        self.ndim = ndim
        self.nhid = nhid
        self.nlay = nlay
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(ndim, nhid, nlay, dropout=pdrop)
        self.attn = attn
        self.attention_model = GlobalAttention(ndim, attn_type)
        self.fc = nn.Linear(nhid, ndim)

    def forward(self, inputs, hiddens, context=None):
        outputs, hiddens = self.rnn(inputs, hiddens)
        if self.attn:
            assert context is not None
            outputs, attentions = self.attention_model(outputs, context)
        outputs = inputs + self.fc(outputs)
        if self.attn:
            return outputs, hiddens, attentions
        else:
            return outputs, hiddens

    def initHidden(self, bsz):
        if self.rnn_type is 'LSTM':
            return (Variable(torch.zeros(self.nlay, bsz, self.nhid)),
                    Variable(torch.zeros(self.nlay, bsz, self.nhid)))
        else:
            return Variable(torch.zeros(self.nlay, bsz, self.nhid))


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = Layers.GraphConvolution(nfeat, nhid)
        self.gc2 = Layers.GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)


class seq2seq(nn.Module):
    def __init__(self, len_in, len_out, ndim, nhid, nlay, pdrop, attn):
        super(seq2seq, self).__init__()
        self.len_in = len_in
        self.len_out = len_out
        self.ndim = ndim
        self.nhid = nhid
        self.nlay = nlay
        self.dropout = nn.Dropout(pdrop)
        self.attn = attn
        self.encoder = Layers.EncoderRNN(ndim, nhid, nlay, pdrop)
        if not attn:
            self.decoder = Layers.DecoderRNN(ndim, nhid, nlay, pdrop)
        else:
            self.decoder = Layers.AttnDecoderRNN(ndim, nhid, nlay, pdrop)

    def forward(self, inputs, targets, cuda=True, teach=False):
        bsz = inputs.size(1)
        len_inp = inputs.size(0)
        len_targ = targets.size(0)
        # encoding
        hid = self.initHidden(bsz, cuda)
        outputs_encoder = Variable(torch.zeros(len_inp, bsz, self.nhid))
        outputs_encoder = outputs_encoder.cuda() if cuda else outputs_encoder
        for ei in range(len_inp):
            outputs_encoder[ei], hid = self.encoder(inputs[ei].unsqueeze(0), hid)

        # decoding
        outputs = []
        attns = []
        dec_inp = inputs[-1].unsqueeze(0)
        for di in range(len_targ):
            if self.attn:
                dec_out, hid, attn = self.decoder(dec_inp, hid, outputs_encoder)
                attns.append(attn)
            else:
                dec_out, hid = self.decoder(dec_inp, hid)
            outputs.append(dec_out)
            if teach and random.random() < 0.5:
                dec_inp = targets[di].unsqueeze(0)
            else:
                dec_inp = dec_out
        if self.attn:
            attns = torch.cat(attns, 0)
        outputs = torch.cat(outputs, 0)
        return (outputs, attns) if self.attn else outputs

    def initHidden(self, bsz, cuda):
        ret = Variable(torch.zeros(self.nlay, bsz, self.nhid))
        return ret.cuda() if cuda else ret


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
