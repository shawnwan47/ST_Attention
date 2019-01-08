import torch.nn as nn


class Framework(nn.Module):
    def __init__(self, model, paradigm, mean, std):
        super().__init__()
        assert paradigm in ['temporal', 'spatial', 'spatialtemporal']
        self.paradigm = paradigm
        self.model = model
        self.mean = mean
        self.std = std

    def pre_forward(self, data, time, weekday):
        if self.paradigm is 'spatialtemporal':
            data = data.unsqueeze(-1)
        elif self.paradigm is 'spatial':
            data = data.transpose(-1, -2)
            time = time[:, -1]
        else:
            pass
        return data, time, weekday

    def post_forward(self, output):
        if self.paradigm is 'spatialtemporal':
            return output.squeeze(-1)
        elif self.paradigm is 'spatial':
            return output.transpose(-1, -2)
        else:
            return output

    def forward(self, data, time, weekday):
        data, time, weekday = self.pre_forward(data, time, weekday)
        output = self.model(data, time, weekday)
        output = self.post_forward(output)
        return output * (self.std + EPS) + self.mean
