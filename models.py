import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import Layers
from Constants import USE_CUDA


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

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)


class EncoderRNN(nn.Module):
    def __init__(self, ndim, nhid, nlay=1, pdrop=0.1):
        super(EncoderRNN, self).__init__()
        self.nhid = nhid
        self.gru = nn.GRU(ndim, nhid, nlay, dropout=pdrop)

    def forward(self, inp, hid):
        return self.gru(inp, hid)

    def initHidden(self, bsz):
        ret = Variable(torch.zeros(1, bsz, self.nhid))
        if USE_CUDA:
            return ret.cuda()
        else:
            return ret


class DecoderRNN(nn.Module):
    def __init__(self, ndim, nhid, nlay=1, pdrop=0.1):
        super(DecoderRNN, self).__init__()
        self.nhid = nhid
        self.gru = nn.GRU(ndim, nhid, nlay, dropout=pdrop)
        self.out = nn.Linear(nhid, ndim)

    def forward(self, inp, hid):
        out, hid = self.gru(inp, hid)
        out = self.out(out)
        return out, hid

    def initHidden(self, bsz):
        ret = Variable(torch.zeros(1, bsz, self.nhid))
        if USE_CUDA:
            return ret.cuda()
        else:
            return ret


class AttnDecoderRNN(nn.Module):
    def __init__(self, ndim, nhid, nlay=1, pdrop=0.1, max_len=24):
        super(AttnDecoderRNN, self).__init__()
        self.ndim = ndim
        self.nhid = nhid

        self.fc_in = nn.Linear(ndim, nhid)
        self.attn_general = nn.Linear(nhid, nhid)
        self.attn_comb = nn.Linear(nhid * 2, nhid)
        self.dropout = nn.Dropout(pdrop)
        self.gru = nn.GRU(nhid, nhid, nlay, dropout=pdrop)
        self.fc_out = nn.Linear(nhid, ndim)

    def forward(self, inp, hid, enc_hids):
        out = self.fc_in(inp)
        out = self.dropout(out)

        out = out.transpose(0, 1)
        enc_hids = enc_hids.transpose(0, 1).transpose(1, 2)
        attn_weights = F.softmax(torch.bmm(self.attn_general(out), enc_hids))
        context = torch.bmm(attn_weights, enc_hids.transpose(1, 2))
        out = self.attn_comb(torch.cat((out, context), -1).transpose(0, 1))

        out, hid = self.gru(out, hid)
        out = inp + self.fc_out(out)  # ResNet
        return out, hid, attn_weights

    def initHidden(self, bsz):
        ret = Variable(torch.zeros(1, bsz, self.nhid))
        if USE_CUDA:
            return ret.cuda()
        else:
            return ret


class GAT(nn.Module):
    def __init__(self, ninp, nout=8, nhid=100, att_heads=[8, 1], att_reduct='concat'):
        self.ninp = ninp
        self.nout = nout
        self.nhid = nhid
        self.att_heads = att_heads
        self.att_reduct = att_reduct

        self.gat1 = 

    def forward(self, inp, adj):
        pass