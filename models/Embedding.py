import torch
import torch.nn as nn


class DayTimeEmbedding(nn.Module):
    def __init__(self, day_count, day_size, time_count, time_size, dropout=0):
        super().__init__()
        self.embedding_day = nn.Embedding(day_count, day_size)
        self.embedding_time = nn.Embedding(time_count, time_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, daytime):
        embed_day = self.embedding_day(daytime[..., 0])
        embed_time = self.embedding_time(daytime[..., 1])
        return self.dropout(torch.cat((embed_time, embed_day), dim=-1))


class STEmbedding(nn.Module):
    def __init__(self, node_count, node_size,
                 day_count, day_size, node_day_size,
                 time_count, time_size, node_time_size, dropout=0):
        super().__init__()
        self.node_count = node_count
        self.node_size = node_size
        self.embedding_day = nn.Embedding(day_count, day_size)
        self.embedding_time = nn.Embedding(time_count, time_size)
        self.embedding_node = nn.Embedding(node_count, node_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_day = nn.Linear(day_size, node_count * node_day_size)
        self.linear_time = nn.Linear(time_size, node_count * node_time_size)

    def forward(self, daytime):
        batch, len_seq, _ = daytime.size()
        # repeat size
        shape = (batch, len_seq, self.node_count, -1)
        # init node
        node = daytime.new_tensor(torch.arange(self.node_count))
        # embedding
        embedded_day = self.dropout(self.embedding_day(daytime[..., 0]))
        embedded_time = self.dropout(self.embedding_time(daytime[..., 1]))
        embedded_day = self.linear_day(embedded_day).view(shape)
        embedded_time = self.linear_time(embedded_time).view(shape)
        embedded_node = self.embedding_node(node).expand(shape)
        embedded = torch.cat((embedded_node, embedded_day, embedded_time), -1)
        return self.dropout(embedded)
