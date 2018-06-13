import torch
import torch.nn as nn
import torch.nn.functional as F


class TempEmbedding(nn.Module):
    def __init__(self, use_time, use_weekday,
                 num_times, time_dim,
                 num_weekdays, weekday_dim,
                 dropout):
        super().__init__()
        self.use_time = use_time
        self.use_weekday = use_weekday
        self.dropout = nn.Dropout(dropout)
        if use_time:
            self.embedding_time = nn.Embedding(num_times, time_dim)
        if use_weekday:
            self.embedding_weekday = nn.Embedding(num_weekdays, weekday_dim)

    def forward(self, data, time, weekday):
        output = [data]
        if self.use_time:
            output.append(self.dropout(self.embedding_time(time)))
        if self.use_weekday:
            output.append(self.dropout(self.embedding_weekday(weekday)))
        return torch.cat(output, -1)


class STEmbedding(nn.Module):
    def __init__(self, use_node, use_time, use_weekday,
                 num_nodes, node_dim,
                 num_times, time_dim,
                 num_weekdays, weekday_dim,
                 input_size, hidden_size, dropout):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_node = use_node
        self.use_time = use_time
        self.use_weekday = use_weekday
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm([num_nodes, hidden_size])
        if use_node:
            self.embedding_node = nn.Embedding(num_nodes, node_dim)
        if use_time:
            self.embedding_time = nn.Embedding(num_times, time_dim)
        if use_weekday:
            self.embedding_weekday = nn.Embedding(num_weekdays, weekday_dim)

    def forward(self, data, time, weekday):
        batch, seq, num_nodes, _ = data.size()
        assert num_nodes == self.num_nodes
        shape = (batch, seq, num_nodes, -1)
        output = [data]
        if self.use_node:
            node = time.new_tensor(torch.arange(num_nodes))
            embedded_node = self.embedding_node(node)
            output.append(self.dropout(embedded_node.expand(shape)))
        if self.use_time:
            embedded_time = self.embedding_time(time).unsqueeze(-2)
            output.append(self.dropout(embedded_time.expand(shape)))
        if self.use_weekday:
            embedded_weekday = self.embedding_weekday(weekday).unsqueeze(-2)
            output.append(self.dropout(embedded_weekday.expand(shape)))
        output = F.relu(self.linear(torch.cat(output, -1)))
        return self.layer_norm(self.dropout(output))
