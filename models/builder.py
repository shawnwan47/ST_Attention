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
from models import Transformer


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


def build_RNN(args):
    embedding = Embedding.build_temp_embedding(args)
    encoder = RNN.build_RNN(args)
    decoder = RNN.build_RNNDecoder(args)


    seq2seq = Seq2Seq.Seq2SeqRNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq


def build_DCRNN(args):
    embedding = Embedding.build_st_embedding(args)

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

    decoder = GCRNN.GCRNNDecoder(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        out_size=args.output_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gc_func=gc_func,
        gc_kwargs=gc_kwargs
    )

    seq2seq = Seq2Seq.Seq2SeqDCRNN(
        embedding=embedding,
        encoder=encoder,
        decoder=decoder,
        history=args.history,
        horizon=args.horizon)
    return seq2seq


def build_GARNN(args):
    embedding = Embedding.build_st_embedding(args)
    if args.model == 'GARNN':
        gc_func = GAT.GraphAttention
        gc_kwargs = {
            'head_count': args.head_count,
            'dropout': args.dropout
        }
    elif args.model == 'GRARNN':
        gc_func = GAT.GraphRelativeAttention
        dist = pt_utils.load_dist(args.dataset, args.num_dists)
        gc_kwargs = {
            'head_count': args.head_count,
            'dropout': args.dropout,
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

    decoder = GCRNN.GARNNDecoder(
        rnn_type=args.rnn_type,
        num_nodes=args.num_nodes,
        size=args.hidden_size,
        out_size=args.output_size,
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
    embedding = Embedding.build_temp_embedding(args)
    encoder = getattr(Transformer, args.model)(
        size=args.hidden_size,
        num_layers=args.num_layers,
        head_count=args.head_count,
        dropout=args.dropout)
    decoder = getattr(Transformer, args.model)(
        size=args.hidden_size,
        out_size=args.output_size,
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
    embedding = Embedding.build_st_embedding(args)
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
