import torch
import torch.nn as nn
from Utils import load_adj


class Bottle(nn.Module):
    def forward(self, inp):
        assert inp.dim() < 5
        if inp.dim() <= 2:
            return super(Bottle, self).forward(inp)
        if inp.dim() == 3:
            out = super(Bottle, self).forward(inp.view(-1, inp.size(2)))
            return out.view(inp.size(0), inp.size(1), -1)
        else:
            out = super(Bottle, self).forward(inp.view(-1, inp.size(3)))
            return out.view(inp.size(0), inp.size(1), inp.size(2), -1)


class LayerNorm(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, dim, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        # HACK. PyTorch is changing behavior
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        out = out.mul(self.a_2.expand_as(out)) \
            + self.b_2.expand_as(out)
        return out


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features=None, bias=True):
        if out_features is None:
            out_features = in_features
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self.adj = load_adj()

    def forward(self, inp):
        if self.adj is not None:
            sparse_weight = self.weight.data.masked_fill(self.adj, 0)
            self.weight = nn.Parameter(sparse_weight)
        return super(SparseLinear, self).forward(inp)


class BottleLinear(Bottle, nn.Linear):
    pass


class BottleSparseLinear(Bottle, SparseLinear):
    pass


class BottleLayerNorm(Bottle, LayerNorm):
    pass
