import numpy as np

import torch
import torch.nn as nn

from lib import pt_utils

from models import Embedding
from models import MLP
from models import RNN
from models import SpatialRNN
from models import DCRNN
from models import Vec2Vec, Seq2Vec, Seq2Seq


def build_model(args):
    history, horizon = args.history, args.horizon
    embedding = build_embedding(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    if args.framework == 'vec2vec':
        model = build_vec2vec(embedding, encoder, decoder)
    elif args.framework == 'seq2vec':
        model = build_seq2vec(embedding, encoder, decoder)
    else:
        model = build_seq2seq(embedding, encoder, decoder, history, horizon)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'{args.path} parameters: {num_params}')
    if args.cuda:
        model.cuda()

    return model

def build_embedding(args):
    if args.paradigm == 'spatialtemporal':
        data_size = args.history if args.framework is 'vec2vec' else 1
        embedding = Embedding.STEmbedding(
            data_size=data_size,
            num_nodes=args.num_nodes,
            num_times=args.num_times,
            bday=args.bday,
            embedding_dim=args.embedding_dim,
            features=args.hidden_size,
            dropout=args.dropout
        )
    else:
        embedding = Embedding.TemporalEmbedding(
            data_size=args.num_nodes,
            num_times=args.num_times,
            bday=args.bday,
            embedding_dim=args.embedding_dim,
            features=args.hidden_size,
            dropout=args.dropout
        )
    return embedding


def build_encoder(args):
    return


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
