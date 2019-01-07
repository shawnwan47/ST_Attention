import torch
import torch.nn as nn

from modules import bias, MLP, ResMLP


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


class HybridEmbedding(nn.Module):
    def __init__(self, num_embeddings, embeddings, embedding_dim, model_dim, dropout):
        assert num_embeddings == embeddings.size(0)
        assert embeddings.dim() == 2
        super().__init__()
        embedding_ = nn.Embedding.from_pretrained(embeddings, freeze=True)
        mlp = MLP(embeddings.size(1), model_dim, dropout)
        self.embedding = Embedding(num_embeddings, embedding_dim, model_dim, dropout)
        self.embedding_ = nn.Sequential(embedding_, mlp)

    def forward(self, input):
        return self.embedding(input) + self.embedding_(input)


class TemporalEmbedding(nn.Module):
    def __init__(self, model_dim, dropout,
                 num_times, time_dim, weekday_dim):
        super().__init__()
        self.embedding_time = Embedding(num_times, time_dim, model_dim, dropout)
        self.embedding_weekday = Embedding(7, weekday_dim, model_dim, dropout)

    def forward(self, time, weekday):
        weekday = self.embedding_weekday(weekday).unsqueeze(-2)
        return self.embedding_time(time) + weekday


class STEmbedding(TemporalEmbedding):
    def __init__(self, model_dim, dropout,
                 num_times, time_dim, weekday_dim,
                 num_nodes, node_dim):
        super().__init__(model_dim, dropout, num_times, time_dim, weekday_dim)
        self.register_buffer('nodes', torch.arange(num_nodes))
        self.embedding_node = Embedding(num_nodes, node_dim, model_dim, dropout)

    def forward(self, time, weekday):
        output = super().forward(time, weekday).unsqueeze(-2)
        output += self.embedding_node(self.nodes)
        return output


class EmbeddingFusion(nn.Module):
    def __init__(self, embedding_data, embedding, model_dim, dropout):
        super().__init__()
        self.embedding_data = embedding_data
        self.embedding = embedding
        self.resmlp = ResMLP(model_dim, dorpout)
        self.nan = bias(model_dim)

    def forward(self, data, time, weekday):
        if data is None:
            output = self.nan
        else:
            output = self.embedding_data(data)
        output += self.embedding(time, weekday)
        return self.resmlp(output)
