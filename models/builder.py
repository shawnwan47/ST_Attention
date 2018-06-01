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
        model = build_seq2seq_rnn(args)
    elif args.model in ['DCRNN', 'GARNN', 'GaARNN']:
        model = build_seq2seq_gcrnn(args)
    elif args.model in ['Transformer']:
        model = build_transformer(args)
    else:
        raise NameError('model {0} unfound!'.format(model))
    return model


def build_temp_embedding(args):
    return Embedding.TempEmbedding(
        use_time=args.use_time, use_weekday=args.use_weekday,
        num_time=args.num_time, time_dim=args.time_dim,
        num_weekday=args.num_weekday, weekday_dim=args.weekday_dim,
        dropout=args.dropout)


def build_st_embedding(args):
    return Embedding.STEmbedding(
        use_node=args.use_node, use_time=args.use_time,
        use_weekday=args.use_weekday,
        num_node=args.num_node, node_dim=args.node_dim,
        num_time=args.num_time, time_dim=args.time_dim,
        num_weekday=args.num_weekday, weekday_dim=args.weekday_dim,
        input_size=args.input_size, hidden_size=args.hidden_size,
        dropout=args.dropout)


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


def build_seq2seq_rnn(args):
    embedding = build_temp_embedding(args)
    encoder = RNN.RNN(
        rnn_type=args.rnn_type,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    if args.model == 'RNN':
        decoder = build_linear(args)
    elif args.model == 'RNNAttn':
        decoder = build_attn_linear(args)

    seq2seq = Seq2Seq.Seq2SeqRNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq


def build_seq2seq_gcrnn(args):
    adj = pt_utils.load_adj(args.dataset)
    if args.cuda:
        adj = adj.cuda()
    embedding = build_st_embedding(args)
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
        num_node=args.num_node,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        func=func,
        gc_kwargs=gc_kwargs
    )

    decoder = Decoder.GraphLinear(
        num_node=args.num_node,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        dropout=args.dropout)

    seq2seq = Seq2Seq.Seq2SeqGCRNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq

def build_transformer(args, adj, hops):
    pass
