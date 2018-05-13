import numpy as np

import torch
import torch.nn as nn

from models import Embedding
from models import RNN
from models import DCRNN


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
    embedding = Embedding.DayTimeEmbedding(
        day_count=args.day_count, day_size=args.day_size,
        time_count=args.time_count, time_size=args.time_size,
        p_dropout=args.p_dropout)
    encoder = RNN.RNN(
        rnn_type=args.rnn_type,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        p_dropout=args.p_dropout)
    if args.model == 'RNN':
        decoder = RNN.RNNDecoder(
            rnn_type=args.rnn_type,
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            p_dropout=args.p_dropout
        )
    else:
        decoder = RNN.RNNAttnDecoder(
            rnn_type=args.rnn_type,
            attn_type=args.attn_type,
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            p_dropout=args.p_dropout
        )
    return RNN.Seq2Seq(args.model, embedding, encoder, decoder,
                       args.past, args.future)


def build_gcrnn(args, adj):
    pass
