import numpy as np

import torch
from constants import EPS
from lib import Loss
from lib import pt_utils


class Interpreter:
    def __init__(self, model, rescaler, loss, cuda):
        model.eval()
        self.model = model
        self.rescaler = rescaler
        self.loss = loss
        if cuda:
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
        else:
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor

    @staticmethod
    def _split_days(data):
        day_size, mod = divmod(data.shape[0], 7)
        assert not mod
        return [data[i * day_size:(i + 1) * day_size] for i in range(7)]

    def eval(self, dataloader):
        targets = []
        outputs = []
        infos = []
        for data, time, day, target in dataloader:
            data = data.type(self.float_type)
            time = time.type(self.long_type)
            day = day.type(self.long_type)
            target = target.type(self.float_type)
            output = self.model(data, time, day)
            if isinstance(output, tuple):
                output, info = output[0], output[1:]
                infos.append([pt_utils.torch_to_numpy(i) for i in info])
                del info
            output = self.rescaler(output)
            targets.append(pt_utils.torch_to_numpy(target))
            outputs.append(pt_utils.torch_to_numpy(output))
            del output
        targets = self._split_days(np.concatenate(targets))
        outputs = self._split_days(np.concatenate(outputs))
        if infos:
            infos = [self._split_days(np.concatenate(info)) for info in zip(*infos)]
        return targets, outputs, infos
