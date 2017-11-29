import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import Layers


class LR(nn.Module):
    def __init__(self, nin, nout):
        super(LR, self).__init__()
        self.linear = nn.Linear(nin, nout)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


class NN(nn.Module):
    def __init__(self, nin, nout, nhid):
        super(NN, self).__init__()
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(nin, nhid)
        self.fc2 = nn.Linear(nhid, nout)

    def forward(self, x):
        out = self.fc1(x.view(x.size(0), -1))
        out = self.relu(self.dropout(out))
        return self.fc2(out)


class RNN(nn.Module):
    def __init__(self, rnn_type, ndim, nhid, nlay):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlay = nlay
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ndim, nhid, nlay, dropout=0.5)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(
                ndim, nhid, nlay,
                nonlinearity=nonlinearity, dropout=0.5)
        self.decoder = nn.Linear(nhid, ndim)

    def forward(self, x):
        weight = next(self.parameters()).data
        bsz = x.size(0)
        if self.rnn_type == 'LSTM':
            hid = (Variable(weight.new(self.nlay, bsz, self.nhid).zero_()),
                   Variable(weight.new(self.nlay, bsz, self.nhid).zero_()))
        else:
            hid = Variable(weight.new(self.nlay, bsz, self.nhid).zero_())
        out, _ = self.rnn(x, hid)
        return self.decoder(out.contiguous().view(-1, self.nhid))


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
    def __init__(self, len_in, len_out, ndim, nhid, nlay=1, pdrop=0.1,
                 attn=False):
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
        len_inp = inputs.size(0)
        bsz = inputs.size(1)
        len_targ = targets.size(0)
        # encoding
        hid = self.initHidden(bsz, cuda)
        enc_outs = Variable(torch.zeros(len_inp, bsz, self.nhid))
        enc_outs = enc_outs.cuda() if cuda else enc_outs
        for ei in range(len_inp):
            enc_outs[ei], hid = self.encoder(inputs[ei].unsqueeze(0), hid)
        outs = []
        attns = []
        dec_inp = inputs[-1].unsqueeze(0)
        for di in range(len_targ):
            if self.attn:
                dec_out, hid, attn = self.decoder(dec_inp, hid, enc_outs)
                attns.append(attn)
            else:
                dec_out, hid = self.decoder(dec_inp, hid)
            outs.append(dec_out)
            if teach and random.random() < 0.5:
                dec_inp = targets[di].unsqueeze(0)
            else:
                dec_inp = dec_out
        if self.attn:
            attns = torch.cat(attns, 0)
        outs = torch.cat(outs, 0)
        return (outs, attns) if self.attn else outs

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
