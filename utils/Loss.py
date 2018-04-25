import torch.nn as nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, mean, scale, od):
        super().__init__()
        self.dim = mean.size(-2) if od is 'OD' else mean.size(-2) // 2
        self.mean = self.masked_select(mean)
        self.scale = self.masked_select(scale)

    def inverse_transform(self, input):
        return input * self.scale + self.mean

    def mae(self, input, target):
        return F.l1_loss(input, target)

    def wape(self, input, target):
        def recover(data):
            return data * self.scale + self.mean
        input, target = recover(input), recover(target)
        return F.l1_loss(input[:,:,0], target[:,:,0]) / target[:,:,0].mean()

    def forward(self, input, target):
        input, target = self.masked_select(input), self.masked_select(target)
        return self.mse(input, target), self.wape(input, target)
