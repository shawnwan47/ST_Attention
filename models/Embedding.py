import torch
import torch.nn as nn


class EmbeddingIn(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, size, dropout):
        super().__init__()
        self.chain = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, size, bias=False)
        )

    def forward(self, input):
        return self.chain(input)


class EmbeddingOut(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.chain = nn.Sequential(
            nn.LayerNorm(size),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.chain(input)


class TempEmbedding(nn.Module):
    def __init__(self, use_time, use_day,
                 num_times, time_dim,
                 num_days, day_dim,
                 num_nodes, size, dropout):
        super().__init__()
        self.size = size
        self.use_time = use_time
        self.use_day = use_day
        self.linear_data = nn.Linear(num_nodes, size)
        self.dropout = nn.Dropout(dropout)
        if use_time:
            self.emb_time = EmbeddingIn(num_times, time_dim, size, dropout)
        if use_day:
            self.emb_day = EmbeddingIn(num_days, day_dim, size, dropout)
        self.emb_out = EmbeddingOut(size, dropout)

    def forward(self, data, time, weekday):
        output = self.dropout(self.linear_data(data))
        if self.use_time:
            output += self.dropout(self.emb_time(time))
        if self.use_day:
            output += self.dropout(self.emb_day(weekday))
        return self.emb_out(output)


class STEmbedding(nn.Module):
    def __init__(self, use_node, use_time, use_day,
                 num_nodes, node_dim,
                 num_times, time_dim,
                 num_days, day_dim,
                 size, dropout):
        super().__init__()
        self.size = size
        self.use_node = use_node
        self.use_time = use_time
        self.use_day = use_day
        self.linear_data = nn.Linear(1, size)
        self.emb_out = EmbeddingOut(size, dropout)
        if use_node:
            self.emb_node = EmbeddingIn(num_nodes, node_dim, size, dropout)
        if use_time:
            self.emb_time = EmbeddingIn(num_times, time_dim, size, dropout)
        if use_day:
            self.emb_day = EmbeddingIn(num_days, day_dim, size, dropout)
        self.register_buffer('nodes', torch.arange(num_nodes))

    def forward(self, data, time, weekday):
        batch, seq, num_nodes, _ = data.size()
        shape = (batch, seq, num_nodes, -1)
        output = self.linear_data(data)
        if self.use_node:
            output += self.emb_node(time.new_tensor(self.nodes))
        if self.use_time:
            output += self.emb_time(time).unsqueeze(-2)
        if self.use_day:
            output += self.emb_day(weekday).unsqueeze(-2)
        output = self.emb_out(output)
        return output


def build_temp_embedding(args):
    return TempEmbedding(
        use_time=args.use_time, use_day=args.use_day,
        num_times=args.num_times, time_dim=args.time_dim,
        num_days=args.num_days, day_dim=args.day_dim,
        num_nodes=args.num_nodes, size=args.hidden_size, dropout=args.dropout)


def build_st_embedding(args):
    return STEmbedding(
        use_node=args.use_node, use_time=args.use_time, use_day=args.use_day,
        num_nodes=args.num_nodes, node_dim=args.node_dim,
        num_times=args.num_times, time_dim=args.time_dim,
        num_days=args.num_days, day_dim=args.day_dim,
        size=args.hidden_size, dropout=args.dropout)
