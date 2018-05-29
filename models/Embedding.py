import torch
import torch.nn as nn
import torch.nn.functional as F


class TempEmbedding(nn.Module):
    def __init__(self, use_time, use_weekday,
                 num_time, time_dim,
                 num_weekday, weekday_dim,
                 dropout):
        super().__init__()
        self.use_time = use_time
        self.use_weekday = use_weekday
        self.dropout = nn.Dropout(dropout)
        if use_time:
            self.embedding_time = nn.Embedding(num_time, time_dim)
        if use_weekday:
            self.embedding_weekday = nn.Embedding(num_weekday, weekday_dim)

    def forward(self, time, weekday):
        output = []
        if self.use_time:
            output.append(self.dropout(self.embedding_time(time)))
        if self.use_weekday:
            output.append(self.dropout(self.embedding_weekday(weekday)))
        return torch.cat(output, -1)


class STEmbedding(nn.Module):
    def __init__(self, use_node, use_time, use_weekday,
                 num_node, node_dim,
                 num_time, time_dim,
                 num_weekday, weekday_dim,
                 dropout):
        super().__init__()
        self.use_node = use_node
        self.use_time = use_time
        self.use_weekday = use_weekday
        self.num_node = num_node
        self.dropout = nn.Dropout(dropout)
        if use_node:
            self.embedding_node = nn.Embedding(num_node, node_dim),
        if use_time:
            self.embedding_time = nn.Embedding(num_time, time_dim)
        if use_weekday:
            self.embedding_weekday = nn.Embedding(num_weekday, weekday_dim)

    def forward(self, time, weekday):
        batch, seq = time.size()
        shape = (batch, seq, self.num_node, -1)
        output = []

        if self.use_node:
            node = time.new_tensor(torch.arange(self.node_count))
            output.append(self.embedding_node(node).expand(shape))
        if self.use_time:
            output.append(self.embedding_time(time).unsqueeze(-2).expand(shape))
        if self.use_weekday:
            output.append(self.embedding_day(weekday).unsqueeze(-2).expand(shape))
        return self.dropout(torch.cat(output, -1))
