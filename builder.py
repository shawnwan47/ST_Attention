import numpy as np
import torch
import torch.nn as nn

from modules import Framework
from modules import MLP
from modules import EmbeddingFusion, TEmbedding, STEmbedding

from models import STTransformer, Transformer, STransformer
from models import RNNSeq2Seq, RNNAttnSeq2Seq
from lib.io import load_mask, load_adj


def build_model(config, mean, std):
    embedding = build_embedding(config)
    if config.model == 'STTransformer':
        model = build_sttransformer(config, embedding)
    elif config.model =='Transformer':
        model = build_transformer(config, embedding)
    elif config.model == 'STransformer':
        model = build_stransformer(config, embedding)
    elif config.model == 'RNN':
        model = build_rnn(config, embedding)
    elif config.model == 'RNNAttn':
        model = build_rnnattn(config, embedding)
    else:
        raise KeyError('Model not implemented!')
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{config.path} parameters: {num_params}')

    model = Framework(model, config.paradigm, mean, std)
    if config.cuda:
        model.cuda()

    return model


def build_embedding(config):
    model_dim = config.model_dim
    dropout = config.dropout

    def build_stembedding():
        return STEmbedding(
            model_dim=model_dim,
            dropout=dropout,
            num_times=config.num_times,
            time_dim=config.time_dim,
            weekday_dim=config.weekday_dim,
            num_nodes=config.num_nodes,
            node_dim=config.node_dim
        )

    def build_tembedding():
        return TEmbedding(
            model_dim=model_dim,
            dropout=dropout,
            num_times=config.num_times,
            time_dim=config.time_dim,
            weekday_dim=config.weekday_dim
        )

    if config.paradigm == 't':
        embedding = build_tembedding()
        data_mlp = MLP(config.num_nodes, model_dim, dropout)
    elif config.paradigm == 's':
        embedding = build_stembedding()
        data_mlp = MLP(config.history, model_dim, dropout)
    elif config.paradigm == 'st':
        embedding = build_stembedding()
        data_mlp = MLP(1, model_dim, dropout)
    else:
        raise KeyError(config.paradigm)
    return EmbeddingFusion(data_mlp, embedding, model_dim, dropout)


def build_sttransformer(config, embedding):
    mask = load_mask(config.dataset) if config.mask else None
    return STTransformer(
        embedding=embedding,
        model_dim=config.model_dim,
        dropout=config.dropout,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        heads=config.heads,
        horizon=config.horizon,
        mask=mask
    )


def build_transformer(config, embedding):
    return Transformer(
        embedding=embedding,
        model_dim=config.model_dim,
        out_dim=config.num_nodes,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        heads=config.heads,
        dropout=config.dropout,
        horizon=config.horizon
    )

def build_stransformer(config, embedding):
    return STransformer(
        embedding=embedding,
        model_dim=config.model_dim,
        out_dim=config.horizon,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout
    )


def build_rnn(config, embedding):
    return RNNSeq2Seq(
        embedding=embedding,
        rnn_type=config.rnn_type,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        out_dim=config.num_nodes,
        dropout=config.dropout,
        horizon=config.horizon
    )


def build_rnnattn(config, embedding):
    return RNNAttnSeq2Seq(
        embedding=embedding,
        rnn_type=config.rnn_type,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        heads=config.heads,
        out_dim=config.num_nodes,
        dropout=config.dropout,
        horizon=config.horizon
    )
