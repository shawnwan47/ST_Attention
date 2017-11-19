import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from layers import GraphConvolution


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


class EmbeddingNN(nn.Module):
    def __init__(self, nin, nout, embed_days, embed_time, nhid):
        super(EmbeddingNN, self).__init__()
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.embedding_days = nn.Embedding(embed_days[0], embed_days[1])
        self.embedding_time = nn.Embedding(embed_time[0], embed_time[1])
        self.fc1 = nn.Linear(nin + embed_days[1] + embed_time[1], nhid)
        self.fc2 = nn.Linear(nhid, nout)

    def forward(self, x, days, time):
        batch_size = x.size(0)
        y = self.dropout(self.embedding_days(days)).view(batch_size, -1)
        z = self.dropout(self.embedding_time(time)).view(batch_size, -1)
        out = torch.cat((x.view(batch_size, -1), y, z), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CNN(nn.Module):
    def __init__(self, width, height, ker, pad):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=ker, padding=pad),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=ker, padding=pad),
            nn.BatchNorm2d(16),
            nn.ReLU())
        nconv = width * height * 16
        self.fc = nn.Linear(nconv, width)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out.view(out.size(0), -1))
        return out


class EmbeddingCNN(nn.Module):
    def __init__(self, width, height, ker, pad, embed_days, embed_time):
        super(EmbeddingCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=ker, padding=pad),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=ker, padding=pad),
            nn.BatchNorm2d(16),
            nn.ReLU())
        nconv = width * height * 16
        self.embedding_days = nn.Embedding(embed_days[0], embed_days[1])
        self.embedding_time = nn.Embedding(embed_time[0], embed_time[1])
        self.dropout = nn.Dropout()
        nin = nconv + embed_days[1] + embed_time[1]
        self.fc = nn.Linear(nin, width)

    def forward(self, x, days, time):
        out = self.layer1(x)
        out = self.layer2(out)
        batch_size = x.size(0)
        y = self.dropout(self.embedding_days(days)).view(batch_size, -1)
        z = self.dropout(self.embedding_time(time)).view(batch_size, -1)
        out = torch.cat((out.view(batch_size, -1), y, z), -1)
        out = self.fc(out)
        return out


class RNN(nn.Module):
    def __init__(self, rnn_type, ndim, nhid, nlayers):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ndim, nhid, nlayers, dropout=0.5, batch_first=True)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(
                ndim, nhid, nlayers,
                nonlinearity=nonlinearity, dropout=0.5, batch_first=True)
        self.decoder = nn.Linear(nhid, ndim)

    def forward(self, x):
        weight = next(self.parameters()).data
        bsz = x.size(0)
        if self.rnn_type == 'LSTM':
            hid = (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                   Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            hid = Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        out, _ = self.rnn(x, hid)
        return self.decoder(out.contiguous().view(-1, self.nhid))


class EmbeddingRNN(nn.Module):
    def __init__(self, rnn_type, ndim, nhid, nlayers, embed_days, embed_time):
        super(EmbeddingRNN, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.embedding_days = nn.Embedding(embed_days[0], embed_days[1])
        self.embedding_time = nn.Embedding(embed_time[0], embed_time[1])
        self.dropout = nn.Dropout()
        nin = ndim + embed_days[1] + embed_time[1]
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                nin, nhid, nlayers, dropout=0.5, batch_first=True)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(
                nin, nhid, nlayers,
                nonlinearity=nonlinearity, dropout=0.5, batch_first=True)
        self.decoder = nn.Linear(nhid, ndim)

    def forward(self, x, days, time):
        weight = next(self.parameters()).data
        bsz = x.size(0)
        if self.rnn_type == 'LSTM':
            hid = (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                   Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            hid = Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        y = self.dropout(self.embedding_days(days))
        z = self.dropout(self.embedding_time(time))
        out = torch.cat((x, y, z), -1)
        out, _ = self.rnn(out, hid)
        return self.decoder(out.contiguous().view(-1, self.nhid))


class CRNN(nn.Module):
    def __init__(self, rnn_type, ndim, nhid, nlayers):
        super(CRNN, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, padding=2),
                nn.BatchNorm1d(8),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2)))
        nin = ndim * 8
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                nin, nhid, nlayers, dropout=0.5, batch_first=True)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(
                ndim, nhid, nlayers,
                nonlinearity=nonlinearity, dropout=0.5, batch_first=True)
        self.decoder = nn.Linear(nhid, ndim)

    def forward(self, x):
        bsz, seq = x.size(0), x.size(1)
        out = x.view(-1, 1, x.size(-1))
        out = self.encoder(out).view(bsz, seq, -1)
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            hid = (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                   Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            hid = Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
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
