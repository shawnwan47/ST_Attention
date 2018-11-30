import numpy as np

import torch
import torch.nn as nn

from lib import pt_utils

from models import Embedding
from models import RNN
from models import DCRNN


def build_model(args):
    if args.framework in ['Seq2Vec', 'Vec2Vec']:
        decoder = MLP(args.hidden_size, args.horizon)
    else:
        decoder = MLP(args.hidden_size, 1)

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


def build_Embedding1D(args):
    return Embedding.Embedding1D(
        num_nodes=args.num_nodes,
        del_time=args.del_time,
        num_times=args.num_times, time_dim=args.time_dim,
        del_day=args.del_day,
        num_days=args.num_days, day_dim=args.day_dim,
        output_size=args.hidden_size, dropout=args.dropout
    )


def build_Embedding2D(args):
    data_size = args.history if args.framework is 'vec2vec' else 1
    return Embedding.STEmbedding(
        data_size=data_size,
        del_nodes=args.del_nodes,
        num_nodes=args.num_nodes, node_dim=args.node_dim,
        del_time=args.del_time,
        num_times=args.num_times, time_dim=args.time_dim,
        del_day=args.del_day,
        num_days=args.num_days, day_dim=args.day_dim,
        output_size=args.hidden_size, dropout=args.dropout
    )


def build_decoder(args):
    return MLP(input_size=args.hidden_size, output_size=args.output_size)


def build_MLP(args):
    assert args.framework is 'vec2vec'
    embedding = build_Embedding1D(args)
    mlp = MLP(input_size=args.hidden_size, args.output_size)
    model = nn.Sequential(embedding, mlp)
    return model


def build_RNN(args):
    assert args.framework in ['seq2seq', 'seq2vec']
    embedding = build_Embedding1D(args)
    decoder = build_decoder(args)
    encoder = RNN.RNN(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    if args.framework is 'seq2seq':
        model = RNN.RNNSeq2Seq(
            embedding=embedding,
            encoder=encoder,
            decoder=decoder,
            history=args.history,
            horizon=args.horizon
        )
    else:
        model = RNN.RNNSeq2Vec(
            embedding=embedding,
            encoder=encoder,
            dcoder=decoder
        )
    return model


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
    embedding = build_Embedding1D(args)
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
