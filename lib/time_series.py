import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeries(Dataset):
    def __init__(self, df, mean, std, history, horizon):
        super().__init__()
        samples = self.gen_samples(df, mean, std, history, horizon)
        self.data, self.time, self.weekday, self.target = samples

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        data = self.data[index]
        time = self.time[index]
        weekday = self.weekday[index]
        target = self.target[index]
        return data, time, weekday, target

    @staticmethod
    def gen_samples(df, mean, std, history, horizon):
        # extract data
        data = (df.values - mean) / (std + 1e-8)
        data[np.isnan(data)] = 0
        _, time = np.unique(df.index.time, return_inverse=True)
        weekday = np.array(df.index.weekday)
        target = df.values

        # split days
        num_days = len(np.unique(df.index.date))
        num_nodes = df.shape[1]

        data = data.reshape(num_days, -1, num_nodes)
        time = time.reshape(num_days, -1)
        weekday = weekday.reshape(num_days, -1)
        target = target.reshape(num_days, -1, num_nodes)

        daily = data.shape[1] - history - horizon
        data = np.stack([data[:, i:i+history] for i in range(daily)], 1)
        time = np.stack([time[:, i:i+history] for i in range(daily)], 1)
        weekday = np.stack([weekday[:, i] for i in range(daily)], 1)
        target = [target[:, i+history:i+history+horizon] for i in range(daily)]
        target = np.stack(target, 1)

        data = data.reshape(-1, history, num_nodes)
        time = time.reshape(-1, history)
        weekday = weekday.reshape(-1)
        target = target.reshape(-1, horizon, num_nodes)

        data = torch.FloatTensor(data)
        time = torch.LongTensor(time)
        weekday = torch.LongTensor(weekday)
        target = torch.FloatTensor(target)
        return data, time, weekday, target
