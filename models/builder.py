import numpy as np

import torch
import torch.nn as nn

from models import RNN, GCRNN


def build_model(args, adj):
    if args.model in ['RNN', 'RNNAttn']:
        model = build_rnn(args)
    elif args.model == 'GCRNN':
        model = build_gcrnn(args, adj)
    elif args.model == 'Transformer':
        model = build_transformer(args)
    return model


def build_rnn(args):
    past, future = args.past, args.future
    embedding = DayTimeEmbedding(args.time_count, args.time_size, args.day_size, args.pdrop)
    encoder = RNN.RNN(
        rnn_type=args.rnn_type,
        nin=args.nin,
        nhid=args.nhid,
        nlayers=args.nlayers,
        pdrop=args.pdrop)
    if args.model == 'RNN':
        decoder = RNN.RNNDecoder(
            rnn_type=args.rnn_type,
            nin=args.nin,
            nout=args.nout,
            nhid=args.nhid,
            nlayers=args.nlayers,
            pdrop=args.pdrop
        )
    else:
        decoder = RNN.RNNAttnDecoder(
            rnn_type=args.rnn_type,
            attn_type=args.attn_type,
            nin=args.nin,
            nout=args.nout,
            nhid=args.nhid,
            nlayers=args.nlayers,
            pdrop=args.pdrop
        )
    return RNN.Seq2Seq(args.model, embedding, encoder, decoder,
                       args.past, args.future)


def build_gcrnn(args, adj):
    pass


def build_transformer(args):
    pass
