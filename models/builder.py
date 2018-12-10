import numpy as np

import torch
import torch.nn as nn

from lib import pt_utils

from models import Embedding
from models import MLP
from models import RNN
from models import SpatialRNN
from models import DCRNN


def build_model(args):
    framework, model = args.framework, args.model
    history, horizon = args.history, args.horizon
    if model == 'SpatialMLP':
        embedding = build_STEmbedding(args)
        mlp = build_SpatialMLP(args)
        model = MLP.MLPVec2Vec(embedding, mlp)
    elif model == 'RNN':
        embedding = build_TemporalEmbedding(args)
        encoder = build_RNN(args)
        if framework == 'seq2vec':
            decoder = build_vector_decoder(args)
            model = RNN.RNNSeq2Vec(embedding, encoder, decoder)
        else:
            decoder = build_RNNDecoder(args)
            model = RNN.RNNSeq2Seq(embedding, encoder, decoder, history, horizon)
    elif model == 'SpatialRNN':
        embedding = build_STEmbedding(args)
        encoder = build_SpatialRNN(args)
        if framework == 'seq2vec':
            decoder = build_vector_decoder(args)
            model = SpatialRNN.SpatialRNNSeq2Vec(embedding, encoder, decoder)
        else:
            decoder = build_SpatialRNNDecoder(args)
            model = SpatialRNN.SpatialRNNSeq2Seq(embedding, encoder, decoder, history, horizon)
    elif model == 'DCRNN':
        embedding = build_STEmbedding(args)
        encoder = build_DCRNN(args)
        if framework == 'seq2vec':
            decoder = MLP.MLP(args.hidden_size, args.output_size)
            model = DCRNN.DCRNNSeq2Vec(embedding, encoder, decoder)
        else:
            decoder = build_DCRNNDecoder(args)
            model = DCRNN.DCRNNSeq2Seq(embedding, encoder, decoder, history, horizon)
    elif model == 'GAT':
        pass
    elif model == 'GATRNN':
        pass
    else:
        raise Exception('model unspecified!')
    return model


def build_TemporalEmbedding(args):
    return Embedding.TemporalEmbedding(
        num_nodes=args.num_nodes,
        del_time=args.del_time,
        num_times=args.num_times, time_dim=args.time_dim,
        del_day=args.del_day,
        num_days=args.num_days, day_dim=args.day_dim,
        output_size=args.hidden_size, dropout=args.dropout
    )


def build_STEmbedding(args):
    data_size = args.history if args.framework is 'vec2vec' else 1
    return Embedding.STEmbedding(
        data_size=data_size,
        del_node=args.del_node,
        num_nodes=args.num_nodes, node_dim=args.node_dim,
        del_time=args.del_time,
        num_times=args.num_times, time_dim=args.time_dim,
        del_day=args.del_day,
        num_days=args.num_days, day_dim=args.day_dim,
        output_size=args.hidden_size, dropout=args.dropout
    )


def build_vector_decoder(args):
    return MLP.MLP(args.hidden_size, args.output_size)


def build_SpatialMLP(args):
    return MLP.MLP(
        input_size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_RNN(args):
    return RNN.RNN(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_RNNDecoder(args):
    return RNN.RNNDecoder(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        output_size=args.num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_SpatialRNN(args):
    return SpatialRNN.SpatialRNN(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        num_nodes=args.num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_SpatialRNNDecoder(args):
    return SpatialRNN.SpatialRNNDecoder(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        num_nodes=args.num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_DCRNN(args):
    adj = pt_utils.load_adj(args.dataset)
    encoder = DCRNN.DCRNN(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        num_layers=args.num_layers,
        adj=adj.cuda() if args.cuda else adj,
        hops=args.hops
    )
    return encoder


def build_DCRNNDecoder(args):
    adj = pt_utils.load_adj(args.dataset)
    decoder = DCRNN.DCRNNDecoder(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        num_layers=args.num_layers,
        adj=adj.cuda() if args.cuda else adj,
        hops=args.hops
    )
    return decoder


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

    encoder = GARNN(
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
    embedding = build_TemporalEmbedding(args)
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
