import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import EPS
from lib.utils import aeq


class IO:
    @staticmethod
    def _get_days(index):
        date_s, date_e = index[0], index[-1]
        days = (date_e - date_s).days + 1
        assert (len(index) % days) == 0
        return days


    def _split_data(self, df, train_ratio=0.7, test_ratio=0.2):
        days = self._get_days(df.index)
        days_train = pd.Timedelta(days=round(days * train_ratio))
        days_test = pd.Timedelta(days=round(days * test_ratio))
        index_train = df.index.date < (df.index[0].date() + days_train)
        index_test = df.index.date >= (df.index[-1].date() - days_test)
        index_valid = (~index_train) & (~index_test)
        return df[index_train], df[index_valid], df[index_test]

    @staticmethod
    def _gen_seq(arr, seq_len, samples):
        seq = np.stack([arr[:, i:i + seq_len] for i in range(samples)], axis=1)
        seq = seq.reshape(-1, seq_len, seq.shape[-1])
        return seq


class TimeSeries(IO):
    def __init__(self, df, start=0, end=24):
        df = df[(df.index.hour >= start) & (df.index.hour < end)]
        self.data_train, self.data_valid, self.data_test = self._split_data(df)
        self.mean = self.data_train.mean().values
        self.std = self.data_train.std().values

    def gen_seq2seq_io(self, df, seq_in_len, seq_out_len, train=False):
        # reshape to days x times x dim
        days = self._get_days(df.index)
        times = len(df.index) // days
        new_shape = (days, times, -1)
        targets = df.values.reshape(new_shape)
        data_cat = self._gen_daytime(df.index).reshape(new_shape)
        data_num = self.scale(df.values).reshape(new_shape)
        data_num[np.isnan(data_num)] = 0.
        # gen seq
        seq_len = seq_in_len + seq_out_len
        data_len = (seq_len - 1) if train else seq_in_len
        samples = times - seq_len + 1
        data_cat = self._gen_seq(data_cat, data_len, samples)
        data_num = self._gen_seq(data_num, data_len, samples)
        targets = self._gen_seq(targets[:, seq_in_len:], seq_out_len, samples)
        return data_num, data_cat, targets

    def scale(self, df):
        return (df - self.mean) / (self.std + EPS)

    @staticmethod
    def _gen_daytime(index):
        day = index.weekday
        _, time = np.unique(index.time, return_inverse=True)
        daytime = np.stack((day, time), -1)
        return daytime



class SparseTimeSeries(IO):
    def __init__(self, df):
        self.data_train, self.data_valid, self.data_test = self._split_data(df)
