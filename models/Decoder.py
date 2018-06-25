import torch.nn as nn


class TempVecLinear(nn.Module):
    def __init__(self, size, num_nodes, horizon):
        super().__init__()
        self.num_nodes
        self.horizon
        self.linear = nn.Linear(size, num_nodes * horizon)

    def forward(self, hidden):
        return self.linear(hidden).view(-1, horizon, num_nodes)


class STSeqLinear(nn.Linear):
    def forward(self, hidden):
        return super().forward(hidden).squeeze(-1)


class STVecLinear(nn.Linear):
    def forward(self, hidden):
        return super().forward(hidden).transpose(-1, -2)


def build_temp_vec(args):
    return TempVecLinear(args.hidden_size, args.num_nodes, args.horizon)


def build_temp_seq(args):
    return nn.Linear(args.hidden_size, args.num_nodes)


def build_st_vec(args):
    return STVecLinear(args.hidden_size, args.horizon)


def build_st_temp(args):
    return STSeqLinear(args.hidden_size, 1)
