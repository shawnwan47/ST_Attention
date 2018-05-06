import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import EPS
from lib.utils import aeq


class IO:
    @staticmethod
    def _split_data(df, val_ratio=0.1, test_ratio=0.2):
        n_sample = df.shape[0]
        n_val = int(round(n_sample * val_ratio))
        n_test = int(round(n_sample * test_ratio))
        n_train = n_sample - n_val - n_test
        return df.iloc[:n_train], df.iloc[n_train:-n_test], df.iloc[-n_test:]

    @staticmethod
    def _gen_seq(arr, length, n_sample):
        return np.stack([arr[i:i+length] for i in range(n_sample)])

    @staticmethod
    def _gen_daytime(date_index):
        day = date_index.weekday
        _, time = np.unique(date_index.time, return_inverse=True)
        daytime = np.stack((day, time), -1)
        return daytime


class TimeSeries(IO):
    def __init__(self, df):
        self.data_train, self.data_valid, self.data_test = self._split_data(df)
        self.mean, self.std = self.data_train.mean(), self.data_train.std()

    def gen_seq2seq_io(self, df, seq_in_len, seq_out_len, train=False):
        seq_len = seq_in_len + seq_out_len
        n_sample = df.shape[0] - seq_len + 1
        data_len = (seq_len - 1) if train else seq_in_len
        daytime = self._gen_daytime(df.index)
        data_cat = self._gen_seq(daytime, data_len, n_sample)
        data_num = self._gen_seq(self.whiten(df).values, data_len, n_sample)
        targets = self._gen_seq(df.values[seq_in_len:], seq_out_len, n_sample)
        data_num[np.isnan(data_num)] = 0
        return data_num, data_cat, targets

    def whiten(self, df):
        return (df - self.mean) / (self.std + EPS)


class SparseTimeSeries(IO):
    def __init__(self, df):
        self.data_train, self.data_valid, self.data_test = self._split_data(df)
