import torch.nn as nn


class DayTimeEmbedding(nn.Module):
    def __init__(self, day_count, day_size, time_count, time_size, p_dropout=0):
        super().__init__()
        self.embedding_day = nn.Embedding(day_count, day_size)
        self.embedding_time = nn.Embedding(time_count, time_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, daytime):
        embed_day = self.embedding_day(daytime[..., 0])
        embed_time = self.embedding_time(daytime[..., 1])
        return self.dropout(torch.cat((embed_time, embed_day), dim=-1))


class STEmbedding(nn.Module):
    def __init__(self, node_count, node_size,
                 day_count, day_size, time_count, time_size, p_dropout=0):
        super().__init__()
        self.node_count = node_count
        self.embedding_day = nn.Embedding(day_count, day_size)
        self.embedding_time = nn.Embedding(time_count, time_size)
        self.embedding_node = nn.Embedding(node_count, node_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, daytime):
        daytime = daytime.unsqueeze(-2)
        node_repeat = list(daytime.size()).pop().append(1)
        time_repeat = [1 for _ in node_repeat]
        time_repeat[-2] = self.node_count
        embed_day = self.embedding_day(daytime[..., 0]).repeat(time_repeat)
        embed_time = self.embedding_time(daytime[..., 1]).repeat(time_repeat)
        node = torch.arange(self.node_count).view(time_repeat)
        embed_node = self.embedding_node(node).repeat(node_repeat)
        embedded = torch.cat((embed_day, embed_time, embed_node), dim=-1)
        return self.dropout(embedded)
