import torch
import torch.nn as nn


class EmbeddingFusion(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, features, dropout):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, features, bias=False)
        )

    def forward(self, input):
        return self.sequential(input)


class TemporalEmbedding(nn.Module):
    def __init__(self, data_size, num_times, bday,
                 embedding_dim, features, dropout):
        super().__init__()
        self.features = features
        self.fc_data = nn.Linear(data_size, features)
        self.embedding_time = EmbeddingFusion(num_times, embedding_dim, features, dropout)
        if not bday:
            self.embedding_day = EmbeddingFusion(7, embedding_dim, features, dropout)
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, data, time, day):
        output = self.fc_data(data)
        output += self.embedding_time(time)
        if hasattr(self, 'embedding_day'):
            output += self.embedding_day(day)
        return self.layer_norm(output)


class STEmbedding(nn.Module):
    def __init__(self, data_size, num_nodes, num_times, bday,
                 embedding_dim, features, dropout):
        super().__init__()
        self.features = features
        self.fc_data = nn.Linear(data_size, features)
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_node = EmbeddingFusion(num_nodes, embedding_dim, features, dropout)
        self.embedding_time = EmbeddingFusion(num_times, embedding_dim, features, dropout)
        if not bday:
            self.embedding_day = EmbeddingFusion(7, embedding_dim, features, dropout)
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, data, time, day):
        output = self.fc_data(data)
        output += self.embedding_node(self.nodes)
        output += self.embedding_time(time)
        if hasattr(self, 'embedding_day'):
            output += self.embedding_day(day)
        return self.layer_norm(output)
