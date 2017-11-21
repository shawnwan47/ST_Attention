import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from Layers import GraphConvolution, SelfAttentiveEncoder
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


class GAN(nn.Module):

    def __init__(self, config):
        super(GAN, self).__init__()
        self.encoder = SelfAttentiveEncoder(config)
        self.fc = nn.Linear(config['nhid'] * config['attention-hops'], config['nfc'])
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.dictionary = config['dictionary']
        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden)
        outp = outp.view(outp.size(0), -1)
        fc = self.tanh(self.fc(self.drop(outp)))
        pred = self.pred(self.drop(fc))
        return pred, attention

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]


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

        self.fc = nn.Linear(ndim, nhid)
        self.attn = nn.Linear(nhid * 2, max_len)
        self.attn_combine = nn.Linear(nhid * 2, nhid)
        self.dropout = nn.Dropout(pdrop)
        self.gru = nn.GRU(nhid, nhid, nlay, dropout=pdrop)
        self.out = nn.Linear(nhid, ndim)

    def forward(self, inp, hid, encoder_outputs):
        out = self.fc(inp)
        out = self.dropout(out)

        attn_weights = F.softmax(self.attn(torch.cat((out, hid), -1)), -1)
        print(attn_weights)
        attn_weights = attn_weights.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_applied = torch.bmm(attn_weights, encoder_outputs).transpose(0, 1)

        out = torch.cat((out, attn_applied), -1)
        out = self.attn_combine(out)

        out, hid = self.gru(out, hid)
        out = self.out(out)  # ResNet
        return out, hid, attn_weights

    def initHidden(self, bsz):
        ret = Variable(torch.zeros(1, bsz, self.nhid))
        if USE_CUDA:
            return ret.cuda()
        else:
            return ret
