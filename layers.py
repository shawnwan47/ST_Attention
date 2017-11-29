import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM()(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class EncoderRNN(nn.Module):
    def __init__(self, ndim, nhid, nlay, pdrop):
        super(EncoderRNN, self).__init__()
        self.nhid = nhid
        self.gru = nn.GRU(ndim, nhid, nlay, dropout=pdrop)

    def forward(self, inp, hid):
        return self.gru(inp, hid)


class DecoderRNN(nn.Module):
    def __init__(self, ndim, nhid, nlay, pdrop):
        super(DecoderRNN, self).__init__()
        self.nhid = nhid
        self.gru = nn.GRU(ndim, nhid, nlay, dropout=pdrop)
        self.fc = nn.Linear(nhid, ndim)

    def forward(self, inp, hid):
        out, hid = self.gru(inp, hid)
        out = inp + self.fc(out)
        return out, hid


class AttnDecoderRNN(nn.Module):

    def __init__(self, ndim, nhid, nlay, pdrop):
        super(AttnDecoderRNN, self).__init__()
        self.ndim = ndim
        self.nhid = nhid

        self.fc_in = nn.Linear(ndim, nhid)
        self.attn_general = nn.Linear(nhid, nhid)
        self.attn_comb = nn.Linear(nhid * 2, nhid)
        self.dropout = nn.Dropout(pdrop)
        self.gru = nn.GRU(nhid, nhid, nlay, dropout=pdrop)
        self.fc_out = nn.Linear(nhid, ndim)

    def forward(self, inp, hid, enc_hids):
        out = self.fc_in(inp)
        out = self.dropout(out)
        # bsz x len x ndim
        out = out.transpose(0, 1)
        enc_hids = enc_hids.transpose(0, 1).transpose(1, 2)
        attn = torch.bmm(self.attn_general(out), enc_hids)
        attn = F.softmax(attn.squeeze(1)).unsqueeze(1)
        context = torch.bmm(attn, enc_hids.transpose(1, 2))
        out = self.attn_comb(torch.cat((out, context), -1).transpose(0, 1))

        out, hid = self.gru(out, hid)
        out = inp + self.fc_out(out)  # ResNet
        return out, hid, attn


class GraphAttention(nn.Module):
    def __init__(self, ninp, nfeat, att_heads=1, nonlinear=False):
        super(GraphAttention, self).__init__()
        self.ninp = ninp
        self.nfeat = nfeat
        self.att_heads = att_heads

        self.kernels = []
        self.att_kernels = []

        for head in range(att_heads):
            self.kernels.append(nn.Linear(ninp, nfeat))
            self.att_kernels.append(nn.Linear(2 * nfeat, 1))

        self.nout = self.nfeat * self.att_heads
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
        for head in range(self.att_heads):
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
