import numpy as np

import torch
import torch.nn as nn

from lib import pt_utils

from models.Framework import Seq2Seq, Seq2Vec
from models.Embedding import Embedding1D, Embedding2D
from models.MLP import MLP
from models.RNN import RNN, RNNDecoder
from models.DCRNN import DCRNN, DCRNNDecoder, DCRNNSeq2Seq



def build_model(args):
    framework, model = args.framework, args.model
    history, horizon = args.history, args.horizon
    if model == 'MLP':
        pass
    elif model == 'RNN':
        embedding = build_Embedding1D(args)
        encoder = build_RNN(args)
        if framework == 'Seq2Vec':
            decoder = MLP(args.hidden_size, args.output_size)
            model = Seq2Vec(embedding, encoder, decoder)
        else:
            decoder = build_RNNDecoder(args)
            model = Seq2Seq(embedding, encoder, decoder, history, horizon)
    elif model == 'DCRNN':
        embedding = build_Embedding2D(args)
        encoder = build_DCRNN(args)
        if framework == 'Seq2Vec':
            decoder = MLP(args.hidden_size, args.output_size)
            model = Seq2Vec(embedding, encoder, decoder)
        else:
            decoder = build_DCRNNDecoder(args)
            model = DCRNNSeq2Seq(embedding, encoder, decoder, history, horizon)
    else:
        raise Exception('model unspecified!')
    return model


def build_Embedding1D(args):
    return Embedding1D(
        num_nodes=args.num_nodes,
        del_time=args.del_time,
        num_times=args.num_times, time_dim=args.time_dim,
        del_day=args.del_day,
        num_days=args.num_days, day_dim=args.day_dim,
        output_size=args.hidden_size, dropout=args.dropout
    )


def build_Embedding2D(args):
    data_size = args.history if args.framework is 'vec2vec' else 1
    return Embedding2D(
        data_size=data_size,
        del_node=args.del_node,
        num_nodes=args.num_nodes, node_dim=args.node_dim,
        del_time=args.del_time,
        num_times=args.num_times, time_dim=args.time_dim,
        del_day=args.del_day,
        num_days=args.num_days, day_dim=args.day_dim,
        output_size=args.hidden_size, dropout=args.dropout
    )


def build_MLP(args):
    assert args.framework is 'vec2vec'
    embedding = build_Embedding1D(args)
    mlp = MLP(input_size=args.hidden_size, output_size=args.output_size)
    model = nn.Sequential(embedding, mlp)
    return model


def build_RNN(args):
    return RNN(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_RNNDecoder(args):
    return RNNDecoder(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_DCRNN(args):
    adj = pt_utils.load_adj(args.dataset)
    encoder = DCRNN(
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
    decoder = DCRNNDecoder(
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
