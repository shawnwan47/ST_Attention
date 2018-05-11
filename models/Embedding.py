import torch.nn as nn


class DayTimeEmbedding(nn.Module):
    def __init__(self, time_count, time_size, day_count, day_size, pdrop=0):
        super().__init__()
        self.embedding_day = nn.Embedding(day_count, day_size)
        self.embedding_time = nn.Embedding(time_count, time_size)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, daytime):
        embed_day = self.embedding_day(daytime[..., 0])
        embed_time = self.embedding_time(daytime[..., 1])
        return self.dropout(torch.cat((embed_time, embed_day), dim=-1))


class STEmbedding(nn.Module):
    def __init__(self, node_count, node_size,
                 day_count, day_size, time_count, time_size, pdrop=0):
        super().__init__()
        self.node_count = node_count
        self.embedding_day = nn.Embedding(day_count, day_size)
        self.embedding_time = nn.Embedding(time_count, time_size)
        self.embedding_node = nn.Embedding(node_count, node_size)

    def forward(self, daytime):
        size = list(daytime.size()[:-1]).insert(-1, self.node_count)
        node = torch.empty(size)
        node[..., :, :]
        embed_day = self.embedding_day(daytime[..., 0])
        embed_time = self.embedding_time(daytime[..., 1])
        node = self.embedding_node(torch.arange(self.node_count))
