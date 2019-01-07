import math
import torch
import torch.nn as nn


class GlobalAttention(nn.Module):
    def __init__(self, size, att_type, dropout=0.1):
        super().__init__()
        assert att_type in ['dot', 'general', 'mlp']
        self.att_type = att_type
        if att_type == 'mlp':
            self.fc_query = nn.Linear(size, size, bias=False)
            self.fc_context = nn.Linear(size, size)
            self.fc_score = nn.Linear(size, 1)
        elif att_type == 'general':
            self.fc_context = nn.Linear(size, size, bias=False)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.fc_out_query = nn.Linear(size, size, False)
        self.fc_out_context = nn.Linear(size, size, bias=att_type=='mlp')

    def forward(self, query, bank, mask=None):
        '''
        query, bank: batch x num x size
        att: batch x num_qry x num_key
        '''
        score = self.score(query, bank)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        context = torch.matmul(attn, bank)
        output = self.fc_out_context(context) + self.fc_out_query(query)
        return output, attn

    def score(self, query, context):
        batch, len_q, size = query.size()
        batch_, len_c, size_ = context.size()
        assert batch == batch_
        assert size == size_
        if self.att_type in ['dot', 'general']:
            if self.att_type == 'general':
                context = self.fc_context(context)
            score = torch.bmm(query, context.transpose(1, 2)) / math.sqrt(size)
        else:
            score_size = (batch, len_q, len_c, size)
            sc1 = self.fc_query(query).unsqueeze(2).expand(score_size)
            sc2 = self.fc_context(context).unsqueeze(1).expand(score_size)
            score = self.fc_score(torch.tanh(sc1 + sc2))
        return score.view(batch, len_q, len_c)
