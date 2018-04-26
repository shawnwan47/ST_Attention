import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.utils import aeq


class TimeSeries:
    def __init__(self, df):
        scaler = StandardScaler().fit(df)
        self.index, self.values = df.index, scaler.transform(df)
        self.mean, self.scale = scaler.mean_, scaler.scale_

    def gen_seq_io(self, start=6, end=23):
        # select time range
        indices = (self.index.hour >= start) & (self.index.hour < end)
        index, ts = self.index[indices], self.values[indices]
        # get day, time
        daytime = self._gen_daytime(index)
        # reshape as daily sequence
        days = (self.index[-1].date() - self.index[0].date()).days + 1
        ts = ts.reshape(days, -1, ts.shape[1])
        daytime = daytime.reshape(days, -1, 2)
        # split data targets
        data_num, data_cat, targets = ts[:-1], daytime[:-1], ts[1:]
        return data_num, data_cat, targets

    def gen_seq2seq_io(self, past, future):
        seq_in_len = past + future - 1
        seq_out_len = future
        daytime = self._gen_daytime(self.index)
        data_num = self._gen_fixed_seq(self.values[:-1], length=seq_in_len)
        data_cat = self._gen_fixed_seq(daytime[:-1], length=seq_in_len)
        targets = self._gen_fixed_seq(self.values[past:], length=seq_out_len)
        aeq(data_num.shape[0], data_cat.shape[0], targets.shape[0])
        return data_num, data_cat, targets

    @staticmethod
    def _gen_fixed_seq(arr, length):
        n_sample = arr.shape[0] - length
        ret = np.array([arr[i:i+length] for i in range(n_sample)])
        return ret

    @staticmethod
    def _gen_daytime(index):
        day = index.weekday
        _, time = np.unique(index.time, return_inverse=True)
        daytime = np.concatenate((day, time), -1)
        return daytime

    @staticmethod
    def train_val_test_split(arr, val_ratio=0.1, test_ratio=0.2):
        n_sample = arr.shape[0]
        n_val = int(round(n_sample * val_ratio))
        n_test = int(round(n_sample * test_ratio))
        n_train = n_sample - n_val - n_test
        return arr[:n_train], arr[n_train:-n_test], arr[-n_test:]
