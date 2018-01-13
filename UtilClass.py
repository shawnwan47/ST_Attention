import torch
import torch.nn as nn


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


class BottleLinear(Bottle, nn.Linear):
    pass


class BottleLayerNorm(Bottle, LayerNorm):
    pass


class ResMLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResMLP, self).__init__()
        self.w_1 = BottleLinear(dim, dim // 2)
        self.w_2 = BottleLinear(dim // 2, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(dim)

    def forward(self, inp):
        out = self.dropout(self.w_2(self.relu(self.w_1(inp))))
        return self.layer_norm(out + inp)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super(MLP, self).__init__()
        hidden_size = (input_size + output_size) // 2
        self.w_1 = BottleLinear(input_size, hidden_size)
        self.w_2 = BottleLinear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        return self.w_2(self.relu(self.dropout(self.w_1(inp))))
