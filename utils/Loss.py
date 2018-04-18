import torch.nn as nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, mean, scale, od):
        super().__init__()
        self.od = od
        self.dim = mean.size(-2) if od is 'OD' else mean.size(-2) // 2
        self.mean = self.masked_select(mean)
        self.scale = self.masked_select(scale)

    def masked_select(self, data):
        assert data.dim() is 3
        if self.od is 'OD':
            ret = data
        elif self.od is 'D':
            ret = data[:, self.dim:]
        elif self.od is 'O':
            ret = data[:, :self.dim]
        return ret

    def mse(self, input, target):
        return F.mse_loss(input, target)

    def wape(self, input, target):
        def recover(data):
            return data * self.scale + self.mean
        input, target = recover(input), recover(target)
        return F.l1_loss(input[:,:,0], target[:,:,0]) / target[:,:,0].mean()

    def forward(self, input, target):
        input, target = self.masked_select(input), self.masked_select(target)
        return self.mse(input, target), self.wape(input, target)
