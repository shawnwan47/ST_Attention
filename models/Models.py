import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.modules import DayTimeEmbedding
from models import Seq2Seq, Tensor2Tensor
import models


def build_model(args):
    if args.framework == 'seq2seq':
        model = build_seq2seq(args)
    if args.framework == 'tensor2tensor':
        model = build_tensor2tensor(args)
    return model


def build_seq2seq(args):
    past, future = args.past, args.future
    embedding = DayTimeEmbedding(args.time_count, args.time_size, args.day_size)
    if args.seq2seq_encoder == 'RNN':
        encoder = models.RNN.RNN(
            rnn_type=args.rnn_type,
            nin=args.nin,
            nhid=args.nhid,
            nlayers=args.nlayers,
            activation=args.activation,
            pdrop=args.pdrop)
        if args.seq2seq_decoder == 'RNN':
            decoder = model.RNN.RNNDecoder(
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
        elif args.seq2seq_decoder == 'RNNAttn':
            decoder = model.RNN.RNNDecoder(
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
