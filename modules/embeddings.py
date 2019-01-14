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


class TEmbedding(nn.Module):
    def __init__(self, model_dim, dropout,
                 num_times, time_dim, weekday_dim):
        super().__init__()
        self.embedding_time = Embedding(num_times, time_dim, model_dim, dropout)
        self.embedding_weekday = Embedding(7, weekday_dim, model_dim, dropout)

    def forward(self, time, weekday):
        emb_time = self.embedding_time(time)
        emb_weekday = self.embedding_weekday(weekday)
        return emb_time + emb_weekday


class STEmbedding(TEmbedding):
    def __init__(self, model_dim, dropout,
                 num_times, time_dim, weekday_dim,
                 num_nodes, node_dim):
        super().__init__(model_dim, dropout, num_times, time_dim, weekday_dim)
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_node = Embedding(num_nodes, node_dim, model_dim, dropout)

    def forward(self, time, weekday):
        s = self.embedding_node(self.nodes)
        t = super().forward(time, weekday)
        return s + t


class EmbeddingFusion(nn.Module):
    def __init__(self, data_mlp, embedding, model_dim, dropout):
        super().__init__()
        self.data_mlp = data_mlp
        self.embedding = embedding
        self.mlp = ResMLP(model_dim, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.register_parameter('nan', bias(model_dim))

    def forward(self, data, time, weekday):
        data = self.nan if data is None else self.data_mlp(data)
        emb = self.embedding(time, weekday)
        return self.layer_norm(self.mlp(data + emb))
