import torch
import torch.nn as nn


class EmbeddingFusion(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_size, dropout):
        super().__init__()
        self.chain = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_size, bias=False)
        )

    def forward(self, input):
        return self.chain(input)


class TemporalEmbedding(nn.Module):
    def __init__(self, num_nodes,
                 del_time, num_times, time_dim,
                 del_day, num_days, day_dim,
                 output_size, dropout):
        super().__init__()
        self.output_size = output_size
        self.del_time = del_time
        self.del_day = del_day
        self.linear_data = nn.Linear(num_nodes, output_size)
        if not del_time:
            self.embedding_time = EmbeddingFusion(num_times, time_dim, output_size, dropout)
        if not del_day:
            self.embedding_day = EmbeddingFusion(num_days, day_dim, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, data, time, weekday):
        output = self.linear_data(data)
        if not self.del_time:
            output += self.embedding_time(time)
        if not self.del_day:
            output += self.embedding_day(weekday)
        return self.layer_norm(output)


class STEmbedding(nn.Module):
    def __init__(self, data_size,
                 del_node, num_nodes, node_dim,
                 del_time, num_times, time_dim,
                 del_day, num_days, day_dim,
                 output_size, dropout):
        super().__init__()
        self.output_size = output_size
        self.del_node = del_node
        self.del_time = del_time
        self.del_day = del_day
        self.linear_data = nn.Linear(data_size, output_size)
        if not del_node:
            self.register_buffer('nodes', torch.arange(num_nodes))
            self.embedding_node = EmbeddingFusion(num_nodes, node_dim, output_size, dropout)
        if not del_time:
            self.embedding_time = EmbeddingFusion(num_times, time_dim, output_size, dropout)
        if not del_day:
            self.embedding_day = EmbeddingFusion(num_days, day_dim, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, data, time, weekday):
        output = self.linear_data(data)
        if not self.del_node:
            output += self.embedding_node(time.new_tensor(self.nodes))
        if not self.del_time:
            output += self.embedding_time(time)
        if not self.del_day:
            output += self.embedding_day(weekday)
        return self.layer_norm(output)
