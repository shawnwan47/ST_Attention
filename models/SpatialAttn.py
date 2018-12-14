import torch
import torch.nn as nn

from lib.utils import aeq
from lib.pt_utils import torch_to_numpy

from models import Attn


class SpatialAttn(nn.Module):
    def __init__(self, in_features, out_features, head_count, dropout, mask=None):
        super().__init__()
        self.attn = Attn.MultiAttn(
            size=in_features,
            head_count=head_count,
            dropout=dropout,
            out_features=out_features
        )
        self.register_buffer('mask', mask)

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x in_features
        '''
        output, attn = self.attn(input, input, input, self.mask)
        return output, attn


class ResSpatialAttn(SpatialAttn):
    def __init__(self, in_features, out_features, head_count, dropout, mask=None):
        super().__init__(*args, **kwargs)
        self.fc_query = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input):
        context, attn = super().forward(input)
        output = self.fc_query(input) + context
        return output


class HighwaySpatialAttn(SpatialAttn):
    def __init__(self, in_features, out_features, head_count, dropout, mask=None):
        super().__init__(in_features, out_features, head_count, dropout, mask)
        self.fc_query = nn.Linear(in_features, out_features)
        self.score_query = nn.Linear(in_features, 1)
        self.score_context = nn.Linear(out_features, 1)
        self.gate = nn.Sigmoid()

    def forward(self, input):
        context, attn = super().forward(input)
        gate = self.gate(self.score_query(input) + self.score_context(context))
        output = (1 - gate) * self.fc_query(query) + gate * context
        return output
