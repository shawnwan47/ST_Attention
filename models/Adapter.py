import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAdapter(nn.Module):
    def __init__(self, data_source,
                 num_node, node_dim,
                 num_time, time_dim,
                 num_weekday, weekday_dim,
                 dropout):
        super().__init__()
        self.data_source = data_source
        self.num_node = num_node
        self.dropout = nn.Dropout(dropout)
        if 'time' in data_source:
            self.embedding_time = nn.Embedding(num_time, time_dim)
        if 'weekday' in data_source:
            self.embedding_weekday = nn.Embedding(7, weekday_dim)
        if 'node' in data_source:
            self.embedding_node = nn.Embedding(num_node, node_dim),

    def forward(self, time, weekday):
        if not self.data_source:
            return

        batch, seq, _ = time.size()
        shape = (batch, seq, self.num_node, -1)
        output = []
        if 'node' in self.data_source:
            node = time.new_tensor(torch.arange(self.node_count))
            output.append(self.embedding_node(node).expand(shape))
        if 'time' in self.data_source:
            output.append(self.embedding_time(time).unsqueeze(-2).expand(shape))
        if 'weekday' in self.data_source:
            output.append(self.embedding_day(weekday).unsqueeze(-2).expand(shape))
        return self.dropout(torch.cat(output, -1))


class VectorAdapter(nn.Module):
    def __init__(self, data_source,
                 num_time, time_dim,
                 num_weekday, weekday_dim,
                 dropout):
        pass
