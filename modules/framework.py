import torch.nn as nn


class Framework(nn.Module):
    def __init__(self, model, paradigm, mean, std):
        super().__init__()
        assert paradigm in ('s', 't', 'st')
        self.paradigm = paradigm
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, data, time, weekday):
        data, time, weekday = self._pre_forward(data, time, weekday)
        output = self.model(data, time, weekday)
        if isinstance(output, tuple):
            assert len(output) == 2
            output, attn = output
            output = self._post_forward(output)
            return output, attn
        return self._post_forward(output)

    def _pre_forward(self, data, time, weekday):
        if self.paradigm == 's':
            data = data.transpose(1, 2)
            time = time[:, [-1]]
            weekday = weekday.unsqueeze(-1)
        elif self.paradigm == 't':
            weekday = weekday.unsqueeze(-1)
        else:
            data = data.unsqueeze(-1)
            time = time.unsqueeze(-1)
            weekday = weekday.unsqueeze(-1).unsqueeze(-1)
        return data, time, weekday

    def _post_forward(self, output):
        if self.paradigm == 'st':
            output = output.squeeze(-1)
        elif self.paradigm == 's':
            output = output.transpose(1, 2)
        else:
            output = output
        return output * (self.std + 1e-8) + self.mean
