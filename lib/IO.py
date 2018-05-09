import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import EPS
from lib.utils import aeq


def get_days(index):
    date_s, date_e = index[0], index[-1]
    days = (date_e - date_s).days + 1
    assert (len(index) % days) == 0, f'{date_s}, {date_e}, {len(index)}, {days}'
    return days


def split_data(data, train_ratio=0.7, test_ratio=0.2):
    if isinstance(data, pd.Series):
        datetimeindex = data.index.levels[0]
        dateindex = data.index.get_level_values(0).date
    else:
        datetimeindex = data.index
        dateindex = data.index.date
    # calculate dates
    days = get_days(datetimeindex)
    days_train = pd.Timedelta(days=round(days * train_ratio))
    days_test = pd.Timedelta(days=round(days * test_ratio))
    date_train = datetimeindex[0].date() + days_train
    date_test = datetimeindex[-1].date() - days_test
    # select data
    index_train = dateindex < date_train
    index_test = dateindex >= date_test
    index_valid = ~(index_train | index_test)
    df_train = data[index_train]
    df_valid = data[index_valid]
    df_test = data[index_test]
    return df_train, df_valid, df_test


class TimeSeries:
    def __init__(self, df, start=0, end=24, seq_len_in=12, seq_len_out=12):
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        df = df[(df.index.hour >= start) & (df.index.hour < end)]
        df_train, df_valid, df_test = split_data(df)
        self.mean, self.std = df_train.mean().values, df_train.std().values
        self.data_train = self._gen_seq2seq_io(df_train, train=True)
        self.data_valid = self._gen_seq2seq_io(df_valid)
        self.data_test = self._gen_seq2seq_io(df_test)


    def _gen_seq2seq_io(self, df, train=False):

        def _gen_daytime(index):
            day = index.weekday
            _, time = np.unique(index.time, return_inverse=True)
            daytime = np.stack((day, time), -1)
            return daytime

        def _gen_seq(arr, seq_len, samples):
            seq = np.stack([arr[:, i:i + seq_len] for i in range(samples)], axis=1)
            seq = seq.reshape(-1, seq_len, seq.shape[-1])
            return seq

        # init data
        data_num = np.nan_to_num(self._scale(df.values))
        data_cat = _gen_daytime(df.index)
        targets = df.values
        # reshape to days x times x dim
        days = get_days(df.index)
        times = len(df.index) // days
        data_num, data_cat, targets = (dat.reshape(days, times, -1)
                                       for dat in (data_num, data_cat, targets))
        # gen seq
        seq_len = self.seq_len_in + self.seq_len_out
        data_len = (seq_len - 1) if train else self.seq_len_in
        samples = times - seq_len + 1
        data_cat = _gen_seq(data_cat, data_len, samples)
        data_num = _gen_seq(data_num, data_len, samples)
        targets = _gen_seq(targets[:, self.seq_len_in:], self.seq_len_out, samples)
        return data_num, data_cat, targets

    def _scale(self, df):
        return (df - self.mean) / (self.std + EPS)


class SparseTimeSeries:
    def __init__(self, ss, start=0, end=24, seq_len_in=12, seq_len_out=12):
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        ss_train, ss_valid, ss_test = split_data(ss)
        self.data_train = self._gen_seq_coo(ss_train)
        self.data_valid = self._gen_seq_coo(ss_valid)
        self.data_test = self._gen_seq_coo(ss_test)

    def _gen_seq_coo(self, ss):
        coos = self.ss_to_coo(ss)
        days = get_days(ss.index.levels[0])
        times = len(coos) // days
        samples = times - self.seq_len_out - self.seq_len_in + 1
        ret = []
        for day in range(days):
            anchor = day * times
            ret.extend([coos[anchor + i:anchor + i + self.seq_len_in]
                        for i in range(samples)])
        return ret

    @staticmethod
    def ss_to_coo(ss):
        coos = []
        for datetime in ss.index.levels[0]:
            val = ss[datetime].values.tolist()
            row = ss[datetime].index.get_level_values(0).tolist()
            col = ss[datetime].index.get_level_values(1).tolist()
            coos.append(((row, col), val))
        return coos
