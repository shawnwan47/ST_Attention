import torch
import torch.nn as nn

from modules.utils import bias, MLP, ResMLP


class TEmbedding(nn.Module):
    def __init__(self, num_times, embedding_dim, dropout):
        super().__init__()
        self.embedding_time = nn.Embedding(num_times, embedding_dim)
        self.embedding_weekday = nn.Embedding(7, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, time, weekday):
        emb_time = self.drop(self.embedding_time(time))
        emb_weekday = self.drop(self.embedding_weekday(weekday))
        return emb_time + emb_weekday


class STEmbedding(TEmbedding):
    def __init__(self, num_times, num_nodes, embedding_dim, dropout):
        super().__init__(num_times, embedding_dim, dropout)
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_node = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, time, weekday):
        t = super().forward(time, weekday)
        s = self.drop(self.embedding_node(self.nodes))
        return s + t


class EmbeddingFusion(nn.Module):
    def __init__(self, data_mlp, embedding, embedding_dim, dropout):
        super().__init__()
        self.data_mlp = data_mlp
        self.embedding = embedding
        self.mlp = ResMLP(embedding_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.register_parameter('nan', bias(embedding_dim))

    def forward(self, data, time, weekday):
        data = self.nan if data is None else self.data_mlp(data)
        emb = self.embedding(time, weekday)
        return self.layer_norm(self.mlp(self.drop(data) + self.drop(emb)))
