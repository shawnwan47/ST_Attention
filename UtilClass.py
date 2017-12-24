import torch
import torch.nn as nn


class Bottle(nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.contiguous().view(size[0], size[1], -1)


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
    def __init__(self, in_features, out_features=None, adj=None, bias=True):
        if out_features is None:
            out_features = in_features
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self.adj = adj

    def forward(self, inp):
        if self.adj is not None:
            sparse_weight = self.weight.masked_fill(self.adj, 0)
            self.weight = nn.Parameter(sparse_weight.data)
        return super(SparseLinear, self).forward(inp)


class BottleLinear(Bottle, nn.Linear):
    pass


class BottleSparseLinear(Bottle, SparseLinear):
    pass


class BottleLayerNorm(Bottle, LayerNorm):
    pass


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Variable whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Variable.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class PointwiseMLP(nn.Module):
    ''' A two-layer Feed-Forward-Network.'''

    def __init__(self, dim, adj=None, dropout=0.1):
        '''
        Args:
            dim(int): the dim of inp for the first-layer of the FFN.
            hidden_size(int): the hidden layer dim of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        '''
        super(PointwiseMLP, self).__init__()
        if adj is None:
            self.w_1 = BottleLinear(dim, dim)
            self.w_2 = BottleLinear(dim, dim)
        else:
            self.w_1 = BottleSparseLinear(dim, dim, adj)
            self.w_2 = BottleSparseLinear(dim, dim, adj)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(dim)

    def forward(self, inp):
        out = self.dropout(self.w_2(self.relu(self.w_1(inp))))
        return self.layer_norm(out + inp)
