import math

import torch
import torch.nn as nn
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
        alphas = alphas.view(size[0], self.att_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas


class GraphAttention(nn.Module):

    def __init__(self, ninp, nfeat, att_heads, att_heads_reduction='concat'):
        super(GraphAttention, self).__init__()
        self.ninp = ninp
        self.nfeat = nfeat
        self.att_heads = att_heads
        self.att_heads_reduction = att_heads_reduction
        self.relu = nn.ReLU()

        self.kernels = []
        self.att_kernels = []

        for head in range(att_heads):
            self.kernels.append(nn.Linear(ninp, nfeat))
            self.att_kernels.append(nn.Linear(2 * nfeat, 1))

        if att_heads_reduction == 'concat':
            self.output_dim = self.nfeat * self.att_heads
        else:
            self.output_dim = self.nfeat

    def forward(self, X, G, A):
        B = X.size()[0]  # Get batch size at runtime
        N = G.size()[0]  # Get number of nodes in the graph at runtime

        outputs = []
        for head in range(self.att_heads):
            kernel = self.kernels[head]
            att_kernel = self.att_kernels[head]

            # Compute inputs to attention network
            linear_transf_X = kernel(X)  # B x F'
            linear_transf_G = kernel(G, kernel)  # N x F'

            # Repeat feature vectors of input: [[1], [2]] becomes [[1], [1], [2], [2]]
            repeated = K.reshape(K.tile(linear_transf_X, [1, N]), (B * N, self.nfeat))  # (BN x F')
            # Tile feature vectors of full graph: [[1], [2]] becomes [[1], [2], [1], [2]]
            tiled = K.tile(linear_transf_G, [B, 1])  # (BN x F')
            # Build combinations
            combinations = K.concatenate([repeated, tiled])  # (BN x 2F')
            combination_slices = K.reshape(combinations, (B, -1, 2 * self.nfeat))  # (B x N x 2F')

            # Attention head
            dense = K.squeeze(K.dot(combination_slices, att_kernel), -1)  # a(Wh_i, Wh_j) in the paper (B x N x 1)
            masked = dense - A  # Masking technique by Vaswani et al., section 2.2 of paper (B x N)
            softmax = K.softmax(masked)  # (B x N)
            dropout = Dropout(0.5)(softmax)  # Apply dropout to normalized attention coefficients (B x N)

            # Linear combination with neighbors' features
            node_features = K.dot(dropout, linear_transf_G)  # (B x F')

            # In the case of concatenation, we compute the activation here (Equation 5)
            if self.att_heads_reduction == 'concat' and self.activation is not None:
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.att_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (B x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0) # (B x F')
            if self.activation is not None:
                # In the case of mean reduction, we compute the activation now
                output = self.activation(output)

        return output
