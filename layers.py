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


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, ninp, nhid, nlayers, att_unit, att_hops):
        super(SelfAttentiveEncoder, self).__init__()
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=0.5)
        self.drop = nn.Dropout(0.5)
        self.ws1 = nn.Linear(nhid, att_unit, bias=False)
        self.ws2 = nn.Linear(att_unit, att_hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.att_hops = att_hops
        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.lstm(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid]
        embeddings = outp.view(-1, size[2])  # [bsz*len, nhid]
        inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        inp = inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        inp = [inp for i in range(self.att_hops)]
        inp = torch.cat(inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(embeddings)))
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.att_hops, size[1])
        return torch.bmm(alphas, outp), alphas


class GraphAttention(nn.Module):
    def __init__(self, ninp, nfeat, att_heads=1, att_reduct='concat'):
        super(GraphAttention, self).__init__()
        self.ninp = ninp
        self.nfeat = nfeat
        self.att_heads = att_heads
        self.att_reduct = att_reduct
        self.relu = nn.ReLU()

        self.kernels = []
        self.att_kernels = []

        for head in range(att_heads):
            self.kernels.append(nn.Linear(ninp, nfeat))
            self.att_kernels.append(nn.Linear(2 * nfeat, 1))

        if att_reduct == 'concat':
            self.output_dim = self.nfeat * self.att_heads
        else:
            self.output_dim = self.nfeat

    def forward(self, inp, adj):
        '''
        inp: bsz x nnode x ninp
        adj: nnode x nnode
        out: bsz x nnode x nfeat
        '''
        bsz, nnode, ninp = inp.size()
        outputs = []
        for head in range(self.att_heads):
            kernel = self.kernels[head]
            att_kernel = self.att_kernels[head]
            out = kernel(inp)

            h = [out for _ in range(nnode)]
            h_i = torch.cat(h, -1).view(bsz, nnode ** 2, -1)
            h_j = torch.cat(h, 1)
            att = att_kernel(torch.cat((h_i, h_j), -1)).view(bsz, nnode, -1)
            att = F.softmax(att[:, adj])
            outputs.append(torch.bmm(att, out))
        if self.att_reduct == 'concat':
            outputs = torch.cat(outputs, -1)
        else:
            outputs = torch.mean(torch.stack(outputs), 0)
        return outputs
