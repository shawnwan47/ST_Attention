import numpy as np

import torch
import torch.nn as nn

from models import Seq2Seq
from models import Decoder
from models import Embedding
from models import RNN
from models import GCRNN
from models import GCN
from models import GAT


def build_model(args, adj):
    if args.model in ['RNN', 'RNNAttn']:
        model = build_rnn(args)
    elif args.model in ['DCRNN', 'GARNN', 'GaARNN']:
        model = build_gcrnn(args, adj)
    elif args.model in ['Transformer', 'ST_Transformer']:
        model = build_transformer(args)
    else:
        raise NameError('model {0} unfound!'.format(model))
    return model


def build_rnn(args):
    embedding = Embedding.DayTimeEmbedding(
        day_count=args.day_count, day_size=args.day_size,
        time_count=args.time_count, time_size=args.time_size,
        dropout=args.dropout)

    encoder = RNN.RNN(
        rnn_type=args.rnn_type,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    if args.model == 'RNN':
        decoder = Decoder.Linear(
            input_size=args.hidden_size,
            output_size=args.output_size,
            dropout=args.dropout)
    return Seq2Seq.Seq2SeqHomo(embedding, encoder, decoder, args.past, args.future)


def build_gcrnn(args, adj):
    embedding = Embedding.STEmbedding(
        node_count=args.node_count, node_size=args.node_size,
        day_count=args.day_count, day_size=args.day_size,
        time_count=args.time_count, time_size=args.time_size,
        dropout=args.dropout)

    if args.model == 'DCRNN':
        func = GCN.DiffusionConvolution
        gc_kwargs = {
            'adj': adj,
            'hops': args.hops,
            'uni': args.uni
        }
    elif args.model == 'GARNN':
        func = GAT.GAT
        gc_kwargs = {
            'head_count': args.head_count,
            'dropout': args.dropout
        }

    encoder = GCRNN.GCRNN(
        rnn_type=args.rnn_type,
        node_count=args.node_count,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        func=func,
        **gc_kwargs
    )

    decoder = Decoder.Linear(args.hidden_size, args.output_size, args.dropout)
    return Seq2Seq.Seq2SeqGCRNN(
        args.model, embedding, encoder, decoder, args.past, args.future)
