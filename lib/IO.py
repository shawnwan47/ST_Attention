import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import EPS
from lib.utils import aeq


def get_days(index):
    date_s, date_e = index[0], index[-1]
    days = (date_e - date_s).days + 1
    assert (len(index) % days) == 0
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


def sample_data(df):
    df_week = df[df.index.weekofyear == (df.index.weekofyear[0] + 1)]
    df_days = [df[df.index.weekday == day] for day in range(7)]
    return df_days



class TimeSeries:
    def __init__(self, df, history=12, horizon=12):
        self.history = history
        self.horizon = horizon
        df_train, df_valid, df_test = split_data(df)
        self.mean, self.std = df_train.mean().values, df_train.std().values
        data_train = self._gen_seq2seq_io(df_train)
        data_valid = self._gen_seq2seq_io(df_valid)
        data_test = self._gen_seq2seq_io(df_test)
        io_samples = [self._gen_seq2seq_io(df) for df in sample_data(df)]
        data_sample = (np.concatenate(data) for data in zip(io_samples))
        self.data = (data_train, data_valid, data_test, data_sample)

    def _gen_seq2seq_io(self, df):

        def _gen_seq(arr, seq_len):
            samples = arr.shape[0] - seq_len + 1
            seq = np.stack([arr[i:i + seq_len] for i in range(samples)], axis=0)
            return seq

        data = df.fillna(method='ffill').fillna(method='bfill')
        assert df.isna().sum().sum() == 0
        data = self._scale(df.values)
        _, time = np.unique(df.index.time, return_inverse=True)
        weekday = df.index.weekday
        data_len = self.history + self.horizon - 1
        data, time, weekday = (_gen_seq(dat[:-1], data_len)
                               for dat in (data, time, weekday))
        targets = _gen_seq(df.values[self.history:], self.horizon)
        aeq(len(data), len(time), len(weekday), len(targets))
        return data, time, weekday, targets

    def _scale(self, df):
        return (df - self.mean) / (self.std + EPS)


class SparseTimeSeries:
    def __init__(self, ts, history=12, horizon=12):
        self.history = history
        self.horizon = horizon
        ss_train, ss_valid, ss_test = split_data(ts)
        self.data_train = self._gen_seq_coo(ss_train)
        self.data_valid = self._gen_seq_coo(ss_valid)
        self.data_test = self._gen_seq_coo(ss_test)

    def _gen_seq_coo(self, ts):
        coos = self.ss_to_coo(ts)
        days = get_days(ts.index.levels[0])
        times = len(coos) // days
        samples = times - self.horizon - self.history + 1
        ret = []
        for day in range(days):
            anchor = day * times
            ret.extend([coos[anchor + i:anchor + i + self.history]
                        for i in range(samples)])
        return ret

    @staticmethod
    def ss_to_coo(ts):
        coos = []
        for datetime in ts.index.levels[0]:
            val = ts[datetime].values.tolist()
            row = ts[datetime].index.get_level_values(0).tolist()
            col = ts[datetime].index.get_level_values(1).tolist()
            coos.append(((row, col), val))
        return coos
