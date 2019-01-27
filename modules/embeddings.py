import math
import torch
import torch.nn as nn

from modules.utils import bias, MLP, ResMLP


class ScalarEmbedding(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.scalar_embedding = MLP(1, model_dim, dropout, bias=False)
        self.nan_embedding = nn.Embedding(2, model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, scalar):
        nan = torch.zeros_like(scalar).long()
        mask = torch.isnan(scalar)
        scalar.masked_fill_(mask, 0.)
        nan.masked_fill_(mask, 1)
        emb_scalar = self.drop(self.scalar_embedding(scalar))
        emb_nan = self.drop(self.nan_embedding(nan)).squeeze(-2)
        return emb_scalar + emb_nan


class VectorEmbedding(nn.Module):
    def __init__(self, vec_dim, model_dim, dropout):
        super().__init__()
        self.vec_embedding = MLP(vec_dim, model_dim, dropout, bias=False)
        self.nan_embedding = nn.Embedding(vec_dim * 2, model_dim)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('nan', torch.arange(vec_dim) * 2)

    def forward(self, vec):
        nan = torch.zeros_like(vec).long()
        mask = torch.isnan(vec)
        vec.masked_fill_(mask, 0.)
        nan.masked_fill_(mask, 1)
        nan = nan + self.nan
        emb_vec = self.drop(self.vec_embedding(vec))
        emb_nan = self.drop(self.nan_embedding(nan)).mean(dim=-2)
        return emb_vec + emb_nan


class TEmbedding(nn.Module):
    def __init__(self, num_times, model_dim, dropout):
        super().__init__()
        self.embedding_time = nn.Embedding(num_times, model_dim)
        self.embedding_weekday = nn.Embedding(7, model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, time, weekday):
        emb_time = self.drop(self.embedding_time(time))
        emb_weekday = self.drop(self.embedding_weekday(weekday))
        return emb_time + emb_weekday


class STEmbedding(TEmbedding):
    def __init__(self, num_times, num_nodes, model_dim, dropout):
        super().__init__(num_times, model_dim, dropout)
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_node = nn.Embedding(num_nodes, model_dim)

    def forward(self, time, weekday):
        t = super().forward(time, weekday)
        s = self.drop(self.embedding_node(self.nodes))
        return s + t


class EmbeddingFusion(nn.Module):
    def __init__(self, embedding_num, embedding_cat, model_dim, dropout):
        super().__init__()
        self.embedding_num = embedding_num
        self.embedding_cat = embedding_cat
        self.mlp = ResMLP(model_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.register_parameter('nan', bias(model_dim))

    def forward(self, data, time, weekday):
        emb_num = self.nan if data is None else self.embedding_num(data)
        emb_cat = self.embedding_cat(time, weekday)
        emb = self.drop(emb_num) + self.drop(emb_cat)
        return self.mlp(emb)
