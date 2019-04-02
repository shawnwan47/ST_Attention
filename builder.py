import numpy as np
import torch
import torch.nn as nn

from modules import Framework
from modules import MLP
from modules import ScalarEmbedding, VectorEmbedding
from modules import EmbeddingFusion, TEmbedding, STEmbedding

from models import IsoMLP, IsoRNN
from models import Transformer, STransformer, STTransformer
from models import RNNSeq2Seq, RNNAttnSeq2Seq
from models import DCRNN
from models import GATRNN

from lib.io import load_adj, load_distant_mask,gen_subsequent_mask


def build_model(config, mean, std):
    embedding = build_embedding(config)
    if config.model == 'IsoMLP':
        model = build_isomlp(config, embedding)
    elif config.model == 'IsoRNN':
        model = build_isornn(config, embedding)
    elif config.model == 'STTransformer':
        model = build_sttransformer(config, embedding)
    elif config.model =='Transformer':
        model = build_transformer(config, embedding)
    elif config.model == 'STransformer':
        model = build_stransformer(config, embedding)
    elif config.model == 'RNN':
        model = build_rnn(config, embedding)
    elif config.model == 'RNNAttn':
        model = build_rnnattn(config, embedding)
    elif config.model == 'DCRNN':
        model = build_dcrnn(config, embedding)
    elif config.model == 'GATRNN':
        model = build_gatrnn(config, embedding)
    else:
        raise KeyError('Model not implemented!')
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{config.path_model} parameters: {num_params}')

    model = Framework(model, config.paradigm, mean, std)
    if config.cuda:
        model.cuda()

    return model


def build_embedding(config):
    if config.paradigm == 't':
        data_mlp = VectorEmbedding(config.num_nodes, config.model_dim, config.dropout)
        # data_mlp = MLP(config.num_nodes, config.model_dim, config.dropout)
        embedding = TEmbedding(
            num_times=config.num_times,
            model_dim=config.model_dim,
            dropout=config.dropout
        )
    else:
        embedding = STEmbedding(
            num_nodes=config.num_nodes,
            num_times=config.num_times,
            model_dim=config.model_dim,
            dropout=config.dropout
        )
        if config.paradigm == 's':
            # data_mlp = MLP(config.history, config.model_dim, config.dropout)
            data_mlp = VectorEmbedding(config.history, config.model_dim, config.dropout)
        else:
            # data_mlp = MLP(1, config.model_dim, config.dropout)
            data_mlp = ScalarEmbedding(config.model_dim, config.dropout)
    return EmbeddingFusion(data_mlp, embedding, config.model_dim, config.dropout)


def build_isomlp(config, embedding):
    return IsoMLP(
        embedding=embedding,
        model_dim=config.model_dim,
        out_dim=config.horizon,
        dropout=config.dropout
    )


def build_isornn(config, embedding):
    return IsoRNN(
        embedding=embedding,
        horizon=config.horizon,
        framework=config.framework,
        rnn_attn=config.rnn_attn,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )


def build_sttransformer(config, embedding):
    return STTransformer(
        embedding=embedding,
        model_dim=config.model_dim,
        dropout=config.dropout,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        heads=config.heads,
        horizon=config.horizon
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


def build_dcrnn(config, embedding):
    return DCRNN(
        embedding=embedding,
        framework=config.framework,
        rnn_attn=config.rnn_attn,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        horizon=config.horizon,
        dropout=config.dropout,
        adj=load_adj(config.dataset),
        hops=config.hops
    )


def build_gatrnn(config, embedding):
    return GATRNN(
        embedding=embedding,
        framework=config.framework,
        rnn_attn=config.rnn_attn,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        horizon=config.horizon,
        heads=config.heads,
        dropout=config.dropout
    )
