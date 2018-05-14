import numpy as np

import torch
import torch.nn as nn

from models import Seq2Seq
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
    return model


def build_rnn(args):
    embedding = Embedding.DayTimeEmbedding(
        day_count=args.day_count, day_size=args.day_size,
        time_count=args.time_count, time_size=args.time_size,
        p_dropout=args.p_dropout)

    kwargs = {
        'rnn_type': args.rnn_type,
        'input_size': args.input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'p_dropout': args.p_dropout
    }

    encoder = RNN.RNN(**kwargs)

    kwargs['output_size'] = args.output_size

    if args.model == 'RNN':
        decoder = RNN.RNNDecoder(**kwargs)
    else:
        kwargs['attn_type'] = args.attn_type
        decoder = RNN.RNNAttnDecoder(**kwargs)
    return Seq2Seq.Seq2SeqRNN(
        args.model, embedding, encoder, decoder, args.past, args.future)


def build_gcrnn(args, adj):
    embedding = Embedding.STEmbedding(
        node_count=args.node_count, node_size=args.node_size,
        day_count=args.day_count, day_size=args.day_size,
        time_count=args.time_count, time_size=args.time_size,
        p_dropout=args.p_dropout)

    if args.model == 'DCRNN':
        gcn_func = GCN.DiffusionConvolution
        gcn_kwargs = {'hops': args.hops, 'reveresed': args.reversed}
    elif args.model == 'GARNN':
        gcn_func = GAT.GAT
        gcn_kwargs = {'head_count': args.head_count,
                      'p_dropout': args.p_dropout}
    elif args.model == 'ResGARNN':
        gcn_func = GAT.ResGAT
        gcn_kwargs = {'head_count': args.head_count,
                      'p_dropout': args.p_dropout}

    encoder = GCRNN.GCRNN(
        rnn_type=args.rnn_type,
        node_count=args.node_count,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        p_dropout=args.p_dropout,
        func=gcn_func, **gcn_kwargs)
    if args.model == 'GCRNN':
        decoder = GCRNN.GCRNN(
            rnn_type=args.rnn_type,
            node_count=args.node_count,
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            p_dropout=args.p_dropout,
            func=gcn_func, **gcn_kwargs)
    else:
        decoder = GCRNN.GCRNN(
            rnn_type=args.rnn_type,
            attn_type=args.attn_type,
            node_count=args.node_count,
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            p_dropout=args.p_dropout,
            func=gcn_func, **gcn_kwargs)
    return Seq2Seq.Seq2SeqGCRNN(
        args.model, embedding, encoder, decoder, args.past, args.future)
