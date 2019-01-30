from random import random

import torch
import torch.nn as nn

from modules import GraphGRUModel


class DCRNN(GraphGRUModel):
    def __init__(self, embedding, framework, rnn_attn,
                 model_dim, num_layers, dropout, horizon, adj, hops):
        super().__init__(
            framework=framework,
            rnn_attn=rnn_attn,
            embedding=embedding,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            func=DiffusionConvolution,
            func_kwargs={'adj': adj, 'hops': hops}
        )


class DiffusionConvolution(nn.Module):
    def __init__(self, in_features, out_features, adj, hops):
        super().__init__()
        self.kernels = hops * 2
        self.linear = nn.Linear(in_features, out_features)
        self.linear_conv = nn.Linear(in_features, out_features * self.kernels)
        diffusion = self.gen_diffusion(adj, hops)
        self.register_buffer('diffusion', diffusion)

    def forward(self, input):
        batch, num_nodes, model_dim = input.size()
        mid = self.linear_conv(input)
        mid = mid.view(batch, num_nodes, self.kernels, -1).transpose(1, 2)
        diffusions = self.diffusion.matmul(mid).sum(dim=1)
        return self.linear(input) + diffusions

    def gen_diffusion(self, adj, hops):
        diffusions = self.gen_diffusions(adj, hops)
        diffusions += self.gen_diffusions(adj.t(), hops)
        return torch.stack(diffusions)

    @staticmethod
    def gen_diffusions(adj, hops):
        adj_norm = adj.div(adj.sum(1).unsqueeze(-1))
        adjs = [adj_norm]
        for _ in range(hops - 1):
            adjs.append(adjs[-1].matmul(adj_norm))
        return adjs
