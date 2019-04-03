import math
import torch
import torch.nn as nn

from modules.utils import bias, MLP, ResMLP


class Embedding(nn.Sequential):
    def __init__(self, num_embeddings, model_dim, dropout):
        embedding_dim = model_dim // 2
        super().__init__(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, model_dim),
            nn.Dropout(dropout)
        )


class ScalarEmbedding(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.embedding_numerical = MLP(1, model_dim, dropout, bias=False)
        self.embedding_nan = nn.Embedding(2, model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, scalar):
        nan = torch.zeros_like(scalar).long()
        mask = torch.isnan(scalar)
        scalar.masked_fill_(mask, 0.)
        nan.masked_fill_(mask, 1)
        emb_scalar = self.drop(self.embedding_numerical(scalar))
        emb_nan = self.drop(self.embedding_nan(nan)).squeeze(-2)
        return emb_scalar + emb_nan


class VectorEmbedding(nn.Module):
    def __init__(self, vec_dim, model_dim, dropout):
        super().__init__()
        self.embedding_vector = MLP(vec_dim, model_dim, dropout, bias=False)
        self.embedding_nan = nn.Embedding(vec_dim * 2, model_dim)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('nan', torch.arange(vec_dim) * 2)

    def forward(self, vec):
        nan = torch.zeros_like(vec).long()
        mask = torch.isnan(vec)
        vec.masked_fill_(mask, 0.)
        nan.masked_fill_(mask, 1)
        nan = nan + self.nan
        emb_vec = self.drop(self.embedding_vector(vec))
        emb_nan = self.drop(self.embedding_nan(nan)).mean(dim=-2)
        return emb_vec + emb_nan


class TEmbedding(nn.Module):
    def __init__(self, num_times, model_dim, dropout):
        super().__init__()
        self.embedding_time = Embedding(num_times, model_dim, dropout)
        self.embedding_weekday = Embedding(7, model_dim, dropout)

    def forward(self, time, weekday):
        emb_time = self.embedding_time(time)
        emb_weekday = self.embedding_weekday(weekday)
        return emb_time + emb_weekday


class STEmbedding(TEmbedding):
    def __init__(self, num_times, num_nodes, model_dim, dropout):
        super().__init__(num_times, model_dim, dropout)
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_node = Embedding(num_nodes, model_dim, dropout)

    def forward(self, time, weekday):
        t = super().forward(time, weekday)
        s = self.embedding_node(self.nodes)
        return s + t


class EmbeddingFusion(nn.Module):
    def __init__(self, embedding_numerical, embedding_categorical,
                 model_dim, dropout):
        super().__init__()
        self.embedding_numerical = embedding_numerical
        self.embedding_categorical = embedding_categorical
        self.resmlp = ResMLP(model_dim, dropout)
        self.ln = nn.LayerNorm(model_dim)
        self.register_parameter('nan', bias(model_dim))

    def forward(self, data, time, weekday):
        emb_num = self.nan if data is None else self.embedding_numerical(data)
        emb_cat = self.embedding_categorical(time, weekday)
        return self.ln(self.resmlp(emb_num + emb_cat))
