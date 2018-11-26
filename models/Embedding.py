import torch
import torch.nn as nn


class EmbeddingLinear(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, size, dropout):
        super().__init__()
        self.chain = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, size, bias=False)
        )

    def forward(self, input):
        return self.chain(input)


class TempEmbedding(nn.Module):
    def __init__(self, del_time, del_day,
                 num_times, time_dim,
                 num_days, day_dim,
                 num_nodes, size, dropout):
        super().__init__()
        self.size = size
        self.del_time = del_time
        self.del_day = del_day
        self.linear_data = nn.Linear(num_nodes, size)
        self.dropout = nn.Dropout(dropout)
        if not del_time:
            self.emb_time = EmbeddingLinear(num_times, time_dim, size, dropout)
        if not del_day:
            self.emb_day = EmbeddingLinear(num_days, day_dim, size, dropout)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, data, time, weekday):
        output = self.dropout(self.linear_data(data))
        if self.del_time:
            output += self.dropout(self.emb_time(time))
        if self.del_day:
            output += self.dropout(self.emb_day(weekday))
        return self.layer_norm(output)


class STEmbedding(nn.Module):
    def __init__(self, del_node, del_time, del_day,
                 num_nodes, node_dim,
                 num_times, time_dim,
                 num_days, day_dim,
                 size, dropout):
        super().__init__()
        self.size = size
        self.del_node = del_node
        self.del_time = del_time
        self.del_day = del_day
        self.linear_data = nn.Linear(1, size)
        self.layer_norm = nn.LayerNorm(size)
        if not del_node:
            self.emb_node = EmbeddingLinear(num_nodes, node_dim, size, dropout)
        if not del_time:
            self.emb_time = EmbeddingLinear(num_times, time_dim, size, dropout)
        if not del_day:
            self.emb_day = EmbeddingLinear(num_days, day_dim, size, dropout)
        self.register_buffer('nodes', torch.arange(num_nodes))

    def forward(self, data, time, weekday):
        batch, seq, num_nodes, _ = data.size()
        shape = (batch, seq, num_nodes, -1)
        output = self.linear_data(data)
        if self.del_node:
            output += self.emb_node(time.new_tensor(self.nodes))
        if self.del_time:
            output += self.emb_time(time).unsqueeze(-2)
        if self.del_day:
            output += self.emb_day(weekday).unsqueeze(-2)
        output = self.layer_norm(output)
        return output
