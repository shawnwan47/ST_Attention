import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import aeq
from UtilClass import *
import Attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim, head=1, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = Attention.MultiHeadAttention(dim, head, dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.mlp = ResMLP(dim, dropout)

    def forward(self, qry, key, val, mask=None):
        out, att = self.attention(qry, key, val, mask)
        out = self.layer_norm(qry + out)
        out = self.mlp(out)
        return out, att.cpu()


class AttentionLayer(nn.Module):
    def __init__(self, dim, map_type='linear', att_type='dot',
                 res=True, mlp=True, dropout=0.5):
        super(AttentionLayer, self).__init__()
        if map_type == 'lin':
            self.map_q = self.map_k = self.map_v = BottleLinear(dim, dim, False)
        elif map_type == 'mlp':
            self.map_q = self.map_k = self.map_v = MLP(dim, dim, dropout)
        elif map_type == 'res':
            self.map_q = self.map_k = self.map_v = ResMLP(dim, dropout)
        self.attention = Attention.Attention(dim, att_type, dropout)
        self.res = res
        self.mlp = mlp
        if res:
            self.layer_norm = BottleLayerNorm(dim)
        if mlp:
            self.resmlp = ResMLP(dim, dropout)

    def forward(self, qry, key, val, mask=None):
        q, k, v = self.map_q(qry), self.map_k(key), self.map_v(val)
        out, att = self.attention(q, k, v, mask)
        if self.res:
            out = self.layer_norm(out + qry)
        if self.mlp:
            out = self.resmlp(out)
        return out, att



class LogisticMixtures(nn.Module):
    def __init__(self, components, categories):
        super(LogisticMixtures, self).__init__()
        self.components = components
        self.categories = categories
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, params):
        '''params: batch x length x num_params'''
        batch, length, num_params = params.size()
        assert num_params % 3 == 0
        params = params.view(-1, self.components, 3)
        prob = self.softmax(params[:, :, [0]])
        mean = params[:, :, [1]]
        inv_std = torch.exp(params[:, :, [2]])
        xs = torch.arange(self.categories) / self.categories * 2 - 1
        xs = xs.expand(batch * length, self.components, self.categories)
        x1, x2 = xs - 1. / self.categories, xs + 1. / self.categories
        x1[:, :, 0] = -1e8
        x2[:, :, -1] = 1e8
        x1, x2 = Variable(x1).cuda(), Variable(x2).cuda()
        cdf1 = self.sigmoid((x1 - mean) * inv_std)
        cdf2 = self.sigmoid((x2 - mean) * inv_std)
        log_pdf = torch.log((prob * (cdf2 - cdf1)).sum(1))
        return log_pdf.view(batch, length, self.categories)
