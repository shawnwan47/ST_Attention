import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeries(Dataset):
    def __init__(self, df, mean, std, history, horizon):
        super().__init__()
        self.data, self.time, self.weekday, self.target = self._df_to_io(df, mean, std)
        self.history = history
        self.horizon = horizon
        self.days = self.data.size(0)
        self.daily_samples = self.data.size(1) - horizon

    def __len__(self):
        return self.days * self.daily_samples

    def __getitem__(self, index):
        d, t = divmod(index, self.daily_samples)
        data = self.data[d, t:t+self.history]
        time = self.time[d, t:t+self.history]
        weekday = self.weekday[d]
        target = self.target[d, t+self.history:t+self.history+self.horizon]
        return data, time, weekday, target

    @staticmethod
    def _df_to_io(df, mean, std):
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
        weekday = weekday.reshape(num_days, -1)[:, 0]
        target = target.reshape(num_days, -1, num_nodes)

        data = torch.FloatTensor(data)
        time = torch.LongTensor(time)
        weekday = torch.LongTensor(weekday)
        target = torch.FloatTensor(target)
        return data, time, weekday, target
