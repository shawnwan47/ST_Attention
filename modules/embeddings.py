import math
import torch
import torch.nn as nn

from modules.utils import bias, ResMLP


class EmbeddingLinear(nn.Sequential):
    def __init__(self, num_embeddings, embedding_dim, model_dim, dropout):
        super().__init__(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, model_dim, bias=False)
        )


class ScalarEmbedding(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(1, model_dim, bias=False)
        self.embedding_nan = nn.Embedding(2, model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, scalar):
        nan = torch.zeros_like(scalar).long()
        mask = torch.isnan(scalar)
        scalar.masked_fill_(mask, 0.)
        nan.masked_fill_(mask, 1)
        emb_scalar = self.fc(scalar)
        emb_nan = self.drop(self.embedding_nan(nan)).squeeze(-2)
        return emb_scalar + emb_nan


class VectorEmbedding(nn.Module):
    def __init__(self, vec_dim, model_dim, dropout):
        super().__init__()
        hidden_size = math.ceil(math.sqrt(model_dim * vec_dim))
        self.fc_0 = nn.Linear(vec_dim, hidden_size, bias=False)
        self.fc_1 = nn.Linear(hidden_size, model_dim, bias=False)
        self.embedding_nan = nn.Embedding(vec_dim + 1, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('nan', torch.arange(vec_dim) + 1)

    def forward(self, vec):
        mask = torch.isnan(vec)
        nan = torch.zeros_like(vec).long() + self.nan
        nan.masked_fill_(~mask, 0)
        vec.masked_fill_(mask, 0.)
        emb_nan = self.embedding_nan(nan).sum(dim=-2)
        f_vec = self.fc_0(vec)
        hidden = f_vec + emb_nan
        return self.fc_1(self.drop(hidden))


class TEmbedding(nn.Module):
    def __init__(self, num_times, time_dim, weekday_dim, model_dim, dropout):
        super().__init__()
        self.embedding_time = EmbeddingLinear(num_times, time_dim, model_dim, dropout)
        self.embedding_weekday = EmbeddingLinear(7, weekday_dim, model_dim, dropout)
        clock_pos = torch.arange(0, math.pi * 2, math.pi * 2 / num_times)
        assert(len(clock_pos) == num_times)
        pos_emb = torch.stack((torch.sin(clock_pos), torch.cos(clock_pos)), 1)
        self.emb_pos_time = nn.Embedding.from_pretrained(pos_emb)
        self.fc_pos_time = nn.Linear(2, model_dim, bias=False)

    def forward(self, time, weekday):
        emb_time = self.embedding_time(time)
        emb_time = emb_time + self.fc_pos_time(self.emb_pos_time(time))
        emb_weekday = self.embedding_weekday(weekday)
        return emb_time + emb_weekday


class STEmbedding(TEmbedding):
    def __init__(self, num_times, time_dim, weekday_dim,
                 num_nodes, node_dim, coordinates,
                 model_dim, dropout):
        super().__init__(num_times, time_dim, weekday_dim, model_dim, dropout)
        assert(num_nodes == len(coordinates))
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.register_buffer('coordinates', coordinates)
        self.embedding_node = EmbeddingLinear(num_nodes, node_dim, model_dim, dropout)
        self.fc_pos_node = nn.Linear(2, model_dim, bias=False)

    def forward(self, time, weekday):
        t = super().forward(time, weekday)
        s = self.embedding_node(self.nodes) + self.fc_pos_node(self.coordinates)
        return s + t


class EmbeddingFusion(nn.Module):
    def __init__(self, mlp_numerical, embedding_categorical,
                 model_dim, dropout):
        super().__init__()
        self.mlp_numerical = mlp_numerical
        self.embedding_categorical = embedding_categorical
        self.resmlp = ResMLP(model_dim, dropout=dropout)
        self.ln = nn.LayerNorm(model_dim)
        self.register_parameter('b', bias(model_dim))

    def forward(self, data, time, weekday):
        embedding = self.embedding_categorical(time, weekday) + self.b
        if data is not None:
            embedding = embedding + self.mlp_numerical(data)
        return self.ln(self.resmlp(embedding))
