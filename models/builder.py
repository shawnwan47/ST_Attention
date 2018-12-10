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
    history, horizon = args.history, args.horizon
    if args.paradigm == 'temporal':
        embedding = build_TemporalEmbedding(args)
        if args.model == 'RNN':
            encoder = build_RNN(args)
            if args.framework == 'seq2vec':
                decoder = build_vector_decoder(args)
                model = RNN.RNNSeq2Vec(embedding, encoder, decoder)
            else:
                decoder = build_RNNDecoder(args)
                model = RNN.RNNSeq2Seq(embedding, encoder, decoder, history, horizon)
        elif args.model == 'DCRNN':
            encoder = build_DCRNN(args)
            if args.framework == 'seq2vec':
                decoder = build_vector_decoder(args)
                model = DCRNN.DCRNNSeq2Vec(embedding, encoder, decoder)
            else:
                decoder = build_DCRNNDecoder(args)
                model = DCRNN.DCRNNSeq2Seq(embedding, encoder, decoder, history, horizon)
    else:
        embedding = build_STEmbedding(args)
    if args.model == 'SpatialMLP':
        embedding = build_STEmbedding(args)
        mlp = build_SpatialMLP(args)
        model = MLP.MLPVec2Vec(embedding, mlp)
    elif args.model == 'SpatialRNN':
        embedding = build_STEmbedding(args)
        encoder = build_SpatialRNN(args)
        if args.framework == 'seq2vec':
            decoder = build_vector_decoder(args)
            model = SpatialRNN.SpatialRNNSeq2Vec(embedding, encoder, decoder)
        else:
            decoder = build_SpatialRNNDecoder(args)
            model = SpatialRNN.SpatialRNNSeq2Seq(embedding, encoder, decoder, history, horizon)
    elif args.model == 'DCRNN':
        embedding = build_STEmbedding(args)
        encoder = build_DCRNN(args)
        if args.framework == 'seq2vec':
            decoder = MLP.MLP(args.hidden_size, args.output_size)
            model = DCRNN.DCRNNSeq2Vec(embedding, encoder, decoder)
        else:
            decoder = build_DCRNNDecoder(args)
            model = DCRNN.DCRNNSeq2Seq(embedding, encoder, decoder, history, horizon)
    elif args.model == 'GAT':
        pass
    elif args.model == 'GATRNN':
        pass
    else:
        raise Exception('model unspecified!')
    return model


def build_TemporalEmbedding(args):
    return Embedding.TemporalEmbedding(
        data_size=args.num_nodes,
        num_times=args.num_times,
        bday=args.bday,
        embedding_dim=args.embedding_dim,
        features=args.hidden_size,
        dropout=args.dropout
    )


def build_STEmbedding(args):
    data_size = args.history if args.framework is 'vec2vec' else 1
    return Embedding.STEmbedding(
        data_size=data_size,
        num_nodes=args.num_nodes,
        num_times=args.num_times,
        bday=args.bday,
        embedding_dim=args.embedding_dim,
        features=args.hidden_size,
        dropout=args.dropout
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
        features=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_RNNDecoder(args):
    return RNN.RNNDecoder(
        rnn_type=args.rnn_type,
        features=args.hidden_size,
        output_size=args.num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout
    )


def build_SpatialRNN(args):
    return SpatialRNN.SpatialRNN(
        rnn_type=args.rnn_type,
        features=args.hidden_size,
        num_nodes=args.num_nodes,
        num_layers=args.num_layers
    )


def build_SpatialRNNDecoder(args):
    return SpatialRNN.SpatialRNNDecoder(
        rnn_type=args.rnn_type,
        features=args.hidden_size,
        num_nodes=args.num_nodes,
        num_layers=args.num_layers
    )


def build_DCRNN(args):
    adj = pt_utils.load_adj(args.dataset)
    encoder = DCRNN.DCRNN(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        features=args.hidden_size,
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
        features=args.hidden_size,
        num_layers=args.num_layers,
        adj=adj.cuda() if args.cuda else adj,
        hops=args.hops
    )
    return decoder
