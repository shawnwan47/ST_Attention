import datetime
import numpy as np
import pandas as pd

from constants import EPS
from lib.utils import aeq


def get_days(index):
    date_s, date_e = index[0], index[-1]
    days = (date_e - date_s).days + 1
    assert (len(index) % days) == 0
    return days


def split_dataset(df, train_ratio=0.7, test_ratio=0.2):
    if isinstance(df, pd.Series):
        datetimeindex = df.index.levels[0]
        dateindex = df.index.get_level_values(0).date
    else:
        datetimeindex = df.index
        dateindex = df.index.date
    # calculate dates
    days = get_days(datetimeindex)
    days_train = pd.Timedelta(days=round(days * train_ratio))
    days_test = pd.Timedelta(days=round(days * test_ratio))
    date_train = datetimeindex[0].date() + days_train
    date_test = datetimeindex[-1].date() - days_test
    # select df
    index_train = dateindex < date_train
    index_test = dateindex >= date_test
    index_valid = ~(index_train | index_test)
    df_train = df[index_train]
    df_valid = df[index_valid]
    df_test = df[index_test]
    return df_train, df_valid, df_test


def df_mean_std(df, method='all'):
    if method is 'all':
        mean = np.nanmean(df.values)
        std = np.nanstd(df.values)
    else:
        mean = df.mean()
        std = df.std()
    return mean, std


def discretize_df(df, num=100):
    values = df.values
    m, M = np.min(values), np.max(values)
    values = np.ceil((values - m) / (M - m) * (num - 1))
    df[:] = values
    return df


def gen_seq(arr, length):
    samples = arr.shape[0] - length + 1
    seq = np.stack([arr[i:i + length] for i in range(samples)], axis=0)
    return seq


def df_to_io(df, history, horizon, mean, std):
    # data, time, weekday
    data = df.fillna(method='ffill').fillna(method='bfill').values
    data = (data - mean) / std
    _, time = np.unique(df.index.time, return_inverse=True)
    weekday = df.index.weekday
    # input sequences
    data_len = history + horizon - 1
    data, time, weekday = (gen_seq(dat[:-1], data_len)
                           for dat in (data, time, weekday))
    # output sequences
    targets = df.values[history:]
    targets = (targets - mean) / std
    targets = gen_seq(targets, horizon)
    aeq(len(data), len(time), len(weekday), len(targets))
    return data, time, weekday, targets


def prepare_dataset(df, history, horizon):
    df_train, df_valid, df_test = split_dataset(df)
    mean, std = df_mean_std(df_train, method='all')
    data_train = df_to_io(df_train, history, horizon, mean, std)
    data_valid = df_to_io(df_valid, history, horizon, mean, std)
    data_test = df_to_io(df_test, history, horizon, mean, std)
    return data_train, data_valid, data_test, mean, std


def prepare_case(df, history, horizon):
    df_week = df[df.index.weekofyear == (df.index.weekofyear[0] + 1)]
    df_days = [df_week[df_week.index.weekday == day] for day in range(7)]
    return (df_to_io(df_day, history, horizon) for df_day in df_days)


class TimeSeries:
    def __init__(self, df, history, horizon):
        self.history = history
        self.horizon = horizon
        df_train, df_valid, df_test = split_dataset(df)
        self.mean, self.std = df_train.mean().values, df_train.std().values
        data_train = self._gen_seq2seq_io(df_train)
        data_valid = self._gen_seq2seq_io(df_valid)
        data_test = self._gen_seq2seq_io(df_test)
        data_case = self._gen_data_case(df)
        self.data = (data_train, data_valid, data_test, data_case)

    def _scale(self, df):
        return (df - self.mean) / (self.std + EPS)

    def _gen_data_case(self, df):
        df_week = df[df.index.weekofyear == (df.index.weekofyear[0] + 1)]
        df_days = [df_week[df_week.index.weekday == day] for day in range(7)]
        io_cases = [self._gen_seq2seq_io(df_day) for df_day in df_days]
        return (np.concatenate(dat) for dat in zip(*io_cases))

    def _gen_seq2seq_io(self, df):

        def _gen_seq(arr, seq_len):
            samples = arr.shape[0] - seq_len + 1
            seq = np.stack([arr[i:i + seq_len] for i in range(samples)], axis=0)
            return seq

        # data
        # data = df.fillna(method='ffill').fillna(method='bfill')
        data = np.nan_to_num(self._scale(df.values))
        # time, weekday
        _, time = np.unique(df.index.time, return_inverse=True)
        weekday = df.index.weekday
        # input sequences
        data_len = self.history + self.horizon - 1
        data, time, weekday = (_gen_seq(dat[:-1], data_len)
                               for dat in (data, time, weekday))
        # output sequences
        targets = _gen_seq(df.values[self.history:], self.horizon)
        aeq(len(data), len(time), len(weekday), len(targets))
        return data, time, weekday, targets


class SparseTimeSeries:
    def __init__(self, ts, history, horizon):
        self.history = history
        self.horizon = horizon
        ss_train, ss_valid, ss_test = split_dataset(ts)
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
