import torch
import torch.nn as nn

from modules.utils import bias, MLP, ResMLP


class ScalarEmbedding(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.linear = nn.Linear(2, model_dim)

    def forward(self, scalar):
        mask = torch.isnan(scalar)
        scalar.masked_fill_(mask, 0.)
        nan = torch.zeros_like(scalar)
        nan.masked_fill_(mask, 1.)
        out = self.linear(torch.cat((scalar, nan), -1))
        return out


class VectorEmbedding(nn.Module):
    def __init__(self, vec_dim, model_dim, dropout):
        super().__init__()
        self.scalar_embeddings = nn.ModuleList([
            ScalarEmbedding(model_dim) for _ in range(vec_dim)
        ])
        self.drop = nn.Dropout(dropout)
        self.linear_out = nn.Linear(model_dim, model_dim)

    def forward(self, vector):
        output = sum(self.drop(embedding(vector[..., [i]]))
                     for i, embedding in enumerate(self.scalar_embeddings))
        return self.linear_out(output)


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
    def __init__(self, vector_embedding, embedding, model_dim, dropout):
        super().__init__()
        self.vector_embedding = vector_embedding
        self.embedding = embedding
        self.mlp = ResMLP(model_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.register_parameter('nan', bias(model_dim))

    def forward(self, data, time, weekday):
        data = self.nan if data is None else self.vector_embedding(data)
        emb = self.embedding(time, weekday)
        return self.mlp(self.drop(data) + self.drop(emb))
