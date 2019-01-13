import torch
import torch.nn as nn

from modules.utils import bias, MLP, ResMLP


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, model_dim, dropout):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, model_dim, bias=False)
        )

    def forward(self, input):
        return self.sequential(input)


class SEmbedding(nn.Module):
    def __init__(self, model_dim, dropout,
                 num_nodes, node_dim,
                 num_times, time_dim, weekday_dim):
        super().__init__()
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_time = Embedding(num_times, time_dim, model_dim, dropout)
        self.embedding_weekday = Embedding(7, weekday_dim, model_dim, dropout)
        self.embedding_node = Embedding(num_nodes, node_dim, model_dim, dropout)

    def forward(self, time, weekday):
        emb_s = self.embedding_node(self.nodes)
        emb_t = self.embedding_time(time) + self.embedding_weekday(weekday)
        return emb_s + emb_t.unsqueeze(-2)


class TEmbedding(nn.Module):
    def __init__(self, model_dim, dropout,
                 num_times, time_dim, weekday_dim):
        super().__init__()
        self.embedding_time = Embedding(num_times, time_dim, model_dim, dropout)
        self.embedding_weekday = Embedding(7, weekday_dim, model_dim, dropout)

    def forward(self, time, weekday):
        emb_weekday = self.embedding_weekday(weekday).unsqueeze(-2)
        emb_time = self.embedding_time(time)
        return emb_weekday + emb_time


class STEmbedding(TEmbedding):
    def __init__(self, model_dim, dropout,
                 num_times, time_dim, weekday_dim,
                 num_nodes, node_dim):
        super().__init__(model_dim, dropout, num_times, time_dim, weekday_dim)
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_node = Embedding(num_nodes, node_dim, model_dim, dropout)

    def forward(self, time, weekday):
        t = super().forward(time, weekday).unsqueeze(-2)
        s = self.embedding_node(self.nodes)
        return s + t


class EmbeddingFusion(nn.Module):
    def __init__(self, data_mlp, embedding, model_dim, dropout):
        super().__init__()
        self.data_mlp = data_mlp
        self.embedding = embedding
        self.resmlp = ResMLP(model_dim, dropout)
        self.register_parameter('nan', bias(model_dim))

    def forward(self, data, time, weekday):
        data = self.nan if data is None else self.data_mlp(data)
        emb = self.embedding(time, weekday)
        return self.resmlp(data + emb)
