import torch
import torch.nn as nn

from modules import MLP, MultiHeadedAttention


class GraphGRUSeq2Vec(nn.Module):
    def __init__(self, embedding, model_dim, num_layers, dropout, horizon, func, func_kwargs):
        super().__init__()
        self.embedding = embedding
        self.rnn = GraphGRU(
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
            func=func,
            func_kwargs=func_kwargs
        )
        self.mlp = MLP(model_dim, horizon, dropout)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        output, _ = self.rnn(input)
        return self.mlp(output[:, [-1]]).transpose(1, -1)


class GraphGRUAttnSeq2Vec(GraphGRUSeq2Vec):
    def __init__(self, embedding, model_dim, num_layers, heads, dropout, horizon, func, func_kwargs):
        super().__init__(
            embedding=embedding,
            model_dim=model_dim,
            horizon=horizon,
            num_layers=num_layers,
            dropout=dropout,
            func=func,
            func_kwargs=func_kwargs
        )
        self.attn = TemporalAttention(
            model_dim=model_dim,
            output_dim=horizon,
            heads=heads,
            dropout=dropout
        )

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        output, _ = self.rnn(input, hidden)
        output_mlp = self.mlp(output[:, [-1]])
        output_context = self.attn(output[:, [-1]], output)
        output = output_mlp + output_context
        return output.transpose(1, -1)


class GraphGRUSeq2Seq(nn.Module):
    def __init__(self, embedding, model_dim, horizon, num_layers, dropout, func, func_kwargs):
        super().__init__()
        self.embedding = embedding
        self.horizon = horizon
        self.encoder = GraphGRU(
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
            func=func,
            func_kwargs=func_kwargs
        )
        self.decoder = GraphGRUDecoder(
            model_dim=model_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            func=func,
            func_kwargs=func_kwargs
        )

    def forward(self, data, time, weekday):
        # encoding
        input = self.embedding(data, time, weekday)
        _, hidden = self.encoder(input)
        # init
        data_i = data[:, [-1]]
        time_i = time[:, [-1]]
        out = []
        # decoding
        for _ in range(self.horizon):
            input_i = self.embedding(data_i.detach(), time_i, weekday)
            res, hidden = self.decoder(input_i, hidden)
            data_i = data_i + res
            time_i = time_i + 1
            out.append(data_i)
        return torch.cat(out, 1)


class GraphGRUAttnSeq2Seq(GraphGRUSeq2Seq):
    def __init__(self, embedding, model_dim, horizon, heads, num_layers, dropout, func, func_kwargs):
        super().__init__(
            embedding=embedding,
            model_dim=model_dim,
            horizon=horizon,
            num_layers=num_layers,
            dropout=dropout,
            func=func,
            func_kwargs=func_kwargs
        )
        self.decoder = GraphGRUAttnDecoder(
            embedding=embedding,
            model_dim=model_dim,
            heads=heads,
            horizon=horizon,
            num_layers=num_layers,
            dropout=dropout,
            func=func,
            func_kwargs=func_kwargs
        )

    def forward(self, data, time, weekday):
        # encoding
        input = self.embedding(data, time, weekday)
        bank, hidden = self.encoder(input)
        # init
        data_i = data[:, [-1]]
        time_i = time[:, [-1]]
        out = []
        # decoding
        for _ in range(self.horizon):
            input_i = self.embedding(data_i.detach(), time_i, weekday)
            res, hidden = self.decoder(input_i, hidden, bank)
            data_i.add_(res)
            time_i.add_(1)
            out.append(data_i)
        return torch.cat(out, 1)


class GraphGRU(nn.Module):
    def __init__(self, model_dim, num_layers, dropout, func, func_kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GraphGRUCell(
                model_dim=model_dim,
                func=func,
                func_kwargs=func_kwargs
            ) for _ in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)
        output = []
        for idx in range(input.size(1)):
            output_i, hidden = self.forward_i(input[:, idx], hidden)
            output.append(output_i)
        output = torch.stack(output, 1)
        return output, hidden

    def forward_i(self, input, hidden):
        hidden_new = []
        for i in range(self.num_layers):
            input = self.drop(self.layers[i](input, hidden[:, i]))
            hidden_new.append(input)
        return input, torch.stack(hidden_new, 1)

    def init_hidden(self, input):
        shape = (input.size(0), self.num_layers, input.size(2), self.model_dim)
        return next(self.parameters()).new_zeros(shape)


class GraphGRUDecoder(GraphGRU):
    def __init__(self, model_dim, output_dim, num_layers, dropout, func, func_kwargs):
        super().__init__(model_dim, num_layers, dropout, func, func_kwargs)
        self.mlp = MLP(model_dim, output_dim, dropout)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        return self.mlp(output), hidden


class GraphGRUAttnDecoder(GraphGRU):
    def __init__(self, model_dim, output_dim, num_layers, dropout, func, func_kwargs):
        super().__init__(model_dim, num_layers, dropout, func, func_kwargs)
        self.attn = TemporalAttention(model_dim, output_dim, heads, dropout)
        self.mlp = MLP(model_dim, output_dim)

    def forward(self, input, hidden, bank):
        output, hidden = super().forward(input, hidden)
        return self.mlp(output) + self.attn(output, bank), hidden


class GraphGRUCell(nn.Module):
    def __init__(self, model_dim, func, func_kwargs):
        super().__init__()
        self.func_i = func(model_dim, model_dim * 3, **func_kwargs)
        self.func_h = func(model_dim, model_dim * 3, **func_kwargs)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input, hidden):
        input_norm = self.layer_norm(input)
        return self.gru(input_norm, hidden)

    def gru(self, input, hidden):
        i_r, i_i, i_n = self.func_i(input).chunk(chunks=3, dim=-1)
        h_r, h_i, h_n = self.func_h(hidden).chunk(chunks=3, dim=-1)
        gate_r = torch.sigmoid(i_r + h_r)
        gate_i = torch.sigmoid(i_i + h_i)
        i_new = torch.tanh(i_n + gate_r * h_n)
        return i_new + gate_i * (hidden - i_new)


class TemporalAttention(MultiHeadedAttention):
    def __init__(self, model_dim, output_dim, heads, dropout):
        super().__init__(model_dim, heads, dropout=dropout, out_dim=output_dim)

    def forward(self, query, bank):
        query = qeury.transpose(1, 2)
        bank = bank.transpose(1, 2)
        return super().forward(query, bank).transpose(1, 2)
