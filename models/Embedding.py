import torch
import torch.nn as nn


class EmbeddingIn(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, size, dropout):
        self.chain = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(embedding_dim, size, bias=False)
        )

    def forward(self, input):
        return self.chain(input)


class EmbeddingOut(nn.Module):
    def __init__(self, size, dropout):
        self.chain = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(size),
            nn.Dropout(inplace=True)
        )

    def forward(self, input):
        return self.chain(input)


class TempEmbedding(nn.Module):
    def __init__(self, use_time, use_weekday,
                 num_times, time_dim,
                 num_days, day_dim,
                 num_nodes, size, dropout):
        super().__init__()
        self.use_time = use_time
        self.use_weekday = use_weekday
        self.linear_data = nn.Linear(num_nodes, size)
        if use_time:
            self.emb_time = EmbeddingIn(num_times, time_dim, size, dropout)
        if use_weekday:
            self.emb_day = EmbeddingIn(num_days, day_dim, size, dropout)
        self.layer_out = EmbeddingOut(size, dropout)

    def forward(self, data, time, weekday):
        output = self.linear_data(data)
        if self.use_time:
            output += self.emb_time(time)
        if self.use_weekday:
            output += self.emb_day(weekday)
        output = self.dropout(self.relu(self.layer_norm(output)))


class STEmbedding(nn.Module):
    def __init__(self, use_node, use_time, use_weekday,
                 num_nodes, node_dim,
                 num_times, time_dim,
                 num_days, day_dim,
                 size, dropout):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_node = use_node
        self.use_time = use_time
        self.use_weekday = use_weekday
        self.linear_data = nn.Linear(1, size)
        self.layer_out = EmbeddingOut(size, dropout)
        if use_node:
            self.emb_node = EmbeddingIn(num_nodes, node_dim, size, dropout)
        if use_time:
            self.emb_time = EmbeddingIn(num_times, time_dim, size, dropout)
        if use_weekday:
            self.emb_day = EmbeddingIn(num_days, day_dim, size, dropout)
        self.register_buffer('nodes', torch.arange(num_nodes))

    def forward(self, data, time, weekday):
        batch, seq, num_nodes, _ = data.size()
        assert num_nodes == self.num_nodes
        shape = (batch, seq, num_nodes, -1)
        output = self.linear_data(data)
        if self.use_node:
            output += self.emb_node(time.new_tensor(self.nodes))
        if self.use_time:
            output += self.emb_time(time).unsqueeze(-2)
        if self.use_weekday:
            output += self.emb_day(weekday).unsqueeze(-2)
        output = self.layer_out(output)
        return output
