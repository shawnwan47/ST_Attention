import numpy as np
import torch
import torch.nn as nn

from modules import MLP
from modules import EmbeddingFusion, TemporalEmbedding, STEmbedding


class Framework(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean, self.std = mean, std

    def forward(self, *data):
        output = self.model(*data)
        return output * (self.std + EPS) + self.mean

def build_model(config, mean, std):
    embedding = build_embedding(config)
    if config.model == 'STTransformer':
        model = build_sttransformer(config, embedding)
    else:
        raise KeyError('Model note implemented!')
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{config.path} parameters: {num_params}')
    framework = Framework(model, mean, std)
    if config.cuda:
        framework.cuda()
    return framework


def build_embedding(config):
    model_dim = config.model_dim
    dropout = config.dropout

    def build_stembedding():
        return STEmbedding(
            model_dim, dropout,
            num_times=config.num_times,
            time_dim=config.time_dim,
            weekday_dim=config.weekday_dim,
            num_nodes=config.num_nodes,
            node_dim=config.node_dim
        )

    def build_tembedding():
        return TEmbedding(
            model_dim, dropout,
            num_times=config.num_times,
            time_dim=config.time_dim,
            weekday_dim=config.weekday_dim
        )

    if config.paradigm == 'temporal':
        embedding = build_tembedding()
        embedding_data = MLP(config.num_nodes, model_dim, dropout)
    elif config.paradigm == 'spatial':
        embedding = build_stembedding()
        embedding_data = MLP(config.history, model_dim, dropout)
    else:
        embedding = build_stembedding()
        embedding_data = MLP(1, model_dim, dropout)
    return EmbeddingFusion(embedding_data, embedding, model_dim, dropout)


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


def build_vector_decoder(config):
    return MLP.MLP(config.hidden_size, config.output_size)


def build_SpatialMLP(config):
    return MLP.MLP(
        input_size=config.hidden_size,
        output_size=config.output_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )


def build_RNN(config):
    return RNN.RNN(
        rnn_type=config.rnn_type,
        features=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )


def build_RNNDecoder(config):
    return RNN.RNNDecoder(
        rnn_type=config.rnn_type,
        features=config.hidden_size,
        output_size=config.num_nodes,
        num_layers=config.num_layers,
        dropout=config.dropout
    )


def build_SpatialRNN(config):
    return SpatialRNN.SpatialRNN(
        rnn_type=config.rnn_type,
        features=config.hidden_size,
        num_nodes=config.num_nodes,
        num_layers=config.num_layers
    )


def build_SpatialRNNDecoder(config):
    return SpatialRNN.SpatialRNNDecoder(
        rnn_type=config.rnn_type,
        features=config.hidden_size,
        num_nodes=config.num_nodes,
        num_layers=config.num_layers
    )


def build_DCRNN(config):
    adj = IO.load_adj(config.dataset)
    encoder = DCRNN.DCRNN(
        rnn_type=config.rnn_type,
        num_nodes=config.num_nodes,
        features=config.hidden_size,
        num_layers=config.num_layers,
        adj=adj.cuda() if config.cuda else adj,
        hops=config.hops
    )
    return encoder


def build_DCRNNDecoder(config):
    adj = IO.load_adj(config.dataset)
    decoder = DCRNN.DCRNNDecoder(
        rnn_type=config.rnn_type,
        num_nodes=config.num_nodes,
        features=config.hidden_size,
        num_layers=config.num_layers,
        adj=adj.cuda() if config.cuda else adj,
        hops=config.hops
    )
    return decoder
