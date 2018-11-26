import numpy as np

import torch
import torch.nn as nn

from lib import pt_utils

from models import Embedding
from models import RNN
from models import DCRNN


def build_model(args):
    if args.model in ['RNN', 'RNNAttn']:
        model = build_RNN(args)
    elif args.model in ['DCRNN']:
        model = build_DCRNN(args)
    elif args.model in ['GARNN', 'GRARNN']:
        model = build_GARNN(args)
    elif args.model in ['Transformer', 'RelativeTransformer']:
        model = build_Transformer(args)
    elif args.model in ['STTransformer', 'RelativeSTTransformer']:
        model = build_STTransformer(args)
    else:
        raise NameError('model {0} unfound!'.format(model))
    return model


def build_temp_embedding(args):
    return Embedding.TempEmbedding(
        del_time=args.del_time, del_day=args.del_day,
        num_times=args.num_times, time_dim=args.time_dim,
        num_days=args.num_days, day_dim=args.day_dim,
        num_nodes=args.num_nodes, size=args.hidden_size, dropout=args.dropout)


def build_st_embedding(args):
    return Embedding.STEmbedding(
        del_node=args.del_node, del_time=args.del_time, del_day=args.del_day,
        num_nodes=args.num_nodes, node_dim=args.node_dim,
        num_times=args.num_times, time_dim=args.time_dim,
        num_days=args.num_days, day_dim=args.day_dim,
        size=args.hidden_size, dropout=args.dropout)


def build_decoder(args):
    pass


def build_RNN(args):
    embedding = build_temp_embedding(args)
    encoder = RNN.RNN(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    decoder = RNN.RNNDecoder(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        output_size=args.num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    seq2seq = RNN.RNNSeq2Seq(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon
    )
    return seq2seq


def build_DCRNN(args):
    embedding = build_st_embedding(args)

    adj = pt_utils.load_adj(args.dataset)

    encoder = DCRNN.DCRNN(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        num_layers=args.num_layers,
        adj=adj.cuda() if args.cuda else adj,
        hops=args.hops
    )

    decoder = DCRNN.DCRNNDecoder(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        num_layers=args.num_layers,
        adj=adj.cuda() if args.cuda else adj,
        hops=args.hops
    )

    seq2seq = DCRNN.DCRNNSeq2Seq(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq


def build_GARNN(args):
    embedding = build_st_embedding(args)
    adj, mask = pt_utils.load_adj_long(args.dataset)
    if args.cuda:
        adj = adj.cuda()
        mask = mask.cuda()
    if args.model == 'GARNN':
        gc_func = GAT.GraphAttention
        gc_kwargs = {
            'head_count': args.head_count,
            'dropout': args.dropout,
            'mask': mask if args.mask else None
        }
    elif args.model == 'GRARNN':
        gc_func = GAT.GraphRelativeAttention
        gc_kwargs = {
            'head_count': args.head_count,
            'dropout': args.dropout,
            'adj': adj,
            'mask': mask if args.mask else None
        }

    encoder = DCRNN.GARNN(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gc_func=gc_func,
        gc_kwargs=gc_kwargs
    )

    decoder = DCRNN.GARNNDecoder(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gc_func=gc_func,
        gc_kwargs=gc_kwargs
    )

    seq2seq = Seq2Seq.Seq2SeqGARNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq


def build_Transformer(args):
    embedding = build_temp_embedding(args)
    encoder = getattr(Transformer, args.model)(
        size=args.hidden_size,
        num_layers=args.num_layers,
        head_count=args.head_count,
        dropout=args.dropout)
    decoder = getattr(Transformer, args.model)(
        size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        head_count=args.head_count,
        dropout=args.dropout)
    decoder = build_linear(args)
    seq2seq = Seq2Seq.Seq2SeqTransformer(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon
    )
    return seq2seq


def build_STTransformer(args):
    embedding = build_st_embedding(args)
    if args.model == 'STTransformer':
        encoder = Transformer.STTransformer(
            size=args.hidden_size,
            num_layers=args.num_layers,
            head_count=args.head_count,
            dropout=args.dropout
        )
    elif args.model == 'RelativeSTTransformer':
        dist = pt_utils.load_dist(args.dataset, args.num_dists)
        encoder = Transformer.RelativeSTTransformer(
            size=args.hidden_size,
            num_layers=args.num_layers,
            head_count=args.head_count,
            dropout=args.dropout,
            dist_s=dist.cuda() if args.cuda else dist
        )
    decoder = build_linear(args)
    seq2seq = Seq2Seq.Seq2SeqSTTransformer(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon
    )
    return seq2seq
