import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import UtilClass


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, nin, nout, bias=True):
        super(GraphConvolution, self).__init__()
        self.nin = nin
        self.nout = nout
        self.linear = UtilClass.BottleLinear(nin, nout)

    def forward(self, inputs, adj):
        '''
        inputs: batch x node x feature
        adj: node x node
        '''
        pass

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.nin) + ' -> ' \
            + str(self.nout) + ')'


class GraphAttention(nn.Module):
    def __init__(self, ninp, nfeat, nheads=1, nonlinear=False):
        super(GraphAttention, self).__init__()
        self.ninp = ninp
        self.nfeat = nfeat
        self.nheads = nheads

        self.kernels = []
        self.att_kernels = []

        for head in range(nheads):
            self.kernels.append(nn.Linear(ninp, nfeat))
            self.att_kernels.append(nn.Linear(2 * nfeat, 1))

        self.nout = self.nfeat * self.nheads
        self.activation = nn.ELU() if nonlinear else None

    def forward(self, inp, adj):
        '''
        inp: bsz x nnode x ninp
        adj: nnode x nnode
        out: bsz x nnode x nfeat
        '''
        bsz, nnode, ninp = inp.size()
        assert adj.size(0) == nnode
        outs, atts = [], []
        for head in range(self.nheads):
            kernel = self.kernels[head]
            att_kernel = self.att_kernels[head]
            out = kernel(inp)

            h = [out for _ in range(nnode)]
            h_i = torch.cat(h, -1).view(bsz, nnode ** 2, -1)
            h_j = torch.cat(h, 1)
            att = att_kernel(torch.cat((h_i, h_j), -1)).view(bsz, nnode, -1)
            att = F.softmax(att[:, adj])
            outs.append(torch.bmm(att, out))
            atts.append(att)
        outs = torch.cat(outs, -1)
        if self.activation:
            outs = self.activation(outs)
        return outs, atts
