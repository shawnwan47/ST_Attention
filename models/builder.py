import numpy as np

import torch
import torch.nn as nn

from lib import pt_utils

from models import Seq2Seq
from models import Decoder
from models import Embedding
from models import RNN
from models import GCRNN
from models import GCN
from models import GAT


def build_model(args):
    if args.model in ['RNN', 'RNNAttn']:
        model = build_RNN(args)
    elif args.model in ['DCRNN']:
        model = build_GCRNN(args)
    elif args.model in ['GARNN', 'GRARNN']:
        model = build_GARNN(args)
    elif args.model in ['ST_Transformer']:
        model = build_ST_Transformer(args)
    else:
        raise NameError('model {0} unfound!'.format(model))
    return model


def build_temp_embedding(args):
    return Embedding.TempEmbedding(
        use_time=args.use_time, use_day=args.use_day,
        num_times=args.num_times, time_dim=args.time_dim,
        num_days=args.num_days, day_dim=args.day_dim,
        num_nodes=args.num_nodes, size=args.hidden_size, dropout=args.dropout)


def build_st_embedding(args):
    return Embedding.STEmbedding(
        use_node=args.use_node, use_time=args.use_time, use_day=args.use_day,
        num_nodes=args.num_nodes, node_dim=args.node_dim,
        num_times=args.num_times, time_dim=args.time_dim,
        num_days=args.num_days, day_dim=args.day_dim,
        size=args.hidden_size, dropout=args.dropout)


def build_linear(args):
    return Decoder.Linear(
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        dropout=args.dropout)


def build_attn_linear(args):
    return Decoder.AttnLinear(
        attn_type=args.attn_type,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        dropout=args.dropout
    )


def build_RNN(args):
    embedding = build_temp_embedding(args)
    encoder = RNN.RNN(
        rnn_type=args.rnn_type,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    decoder = build_linear(args)

    seq2seq = Seq2Seq.Seq2SeqRNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq


def build_GCRNN(args):
    embedding = build_st_embedding(args)
    adj = pt_utils.load_adj(args.dataset)
    gc_func = GCN.DiffusionConvolution
    gc_kwargs = {
        'adj': adj.cuda() if args.cuda else adj,
        'hops': args.hops
    }

    encoder = GCRNN.GCRNN(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gc_func=gc_func,
        gc_kwargs=gc_kwargs
    )

    decoder = build_linear(args)

    seq2seq = Seq2Seq.Seq2SeqDCRNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq


def build_GARNN(args):
    embedding = build_st_embedding(args)
    if args.model == 'GARNN':
        gc_func = GAT.GraphAttention
        gc_kwargs = {
            'head_count': args.head_count,
            'dropout': args.dropout
        }
    elif args.model == 'GRARNN':
        gc_func = GAT.GraphRelativeAttention
        dist = pt_utils.load_dist(args.dataset, args.num_node_dists)
        gc_kwargs = {
            'head_count': args.head_count,
            'dropout': args.dropout,
            'num_dists': args.num_node_dists,
            'dist': dist.cuda() if args.cuda else dist
        }

    encoder = GCRNN.GARNN(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gc_func=gc_func,
        gc_kwargs=gc_kwargs
    )

    decoder = build_linear(args)

    seq2seq = Seq2Seq.Seq2SeqGARNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq
