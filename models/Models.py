import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import models
from models import Seq2Seq, Tensor2Tensor, RNN


class DayTimeEmbedding(nn.Module):
    def __init__(self, num_time, time_size, day_size, pdrop=0):
        super().__init__()
        self.embedding_day = nn.Embedding(7, day_size)
        self.embedding_time = nn.Embedding(num_time, time_size)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, data_cat):
        embedded_day = self.embedding_day(data_cat[:, :, 0])
        embedded_time = self.embedding_time(data_cat[:, :, 1])
        return self.dropout(torch.cat((embedded_time, embedded_day), dim=-1))


def build_model(args):
    if args.framework == 'seq2seq':
        model = build_seq2seq(args)
    if args.framework == 'tensor2tensor':
        model = build_tensor2tensor(args)
    return model


def build_seq2seq(args):
    past, future = args.past, args.future
    embedding = DayTimeEmbedding(args.time_count, args.time_size, args.day_size)
    if args.model in ['RNN', 'RNNAttn']:
        encoder = RNN.RNN(
            rnn_type=args.rnn_type,
            nin=args.nin,
            nhid=args.nhid,
            nlayers=args.nlayers,
            activation=args.activation,
            pdrop=args.pdrop)
        if args.model == 'RNN':
            decoder = RNN.RNNDecoder(
                rnn_type=args.rnn_type,
                nin=args.nin,
                nout=args.nout,
                nhid=args.nhid,
                nlayers=args.nlayers,
                activation=args.activation,
                pdrop=args.pdrop
            )
            return Seq2Seq.Seq2SeqRNN(embedding, encoder, decoder,
                                      args.past, args.future)
        else:
            decoder = RNN.RNNDecoder(
                rnn_type=args.rnn_type,
                attn_type=args.attn_type,
                nin=args.nin,
                nout=args.nout,
                nhid=args.nhid,
                nlayers=args.nlayers,
                activation=args.activation,
                pdrop=args.pdrop
            )
            return Seq2Seq.Seq2SeqRNNAttn(embedding, encoder, decoder,
                                          args.past, args.future)
