import datetime
import numpy as np
import pandas as pd

from constants import EPS
from lib.utils import aeq



def prepare_dataset(df, history, horizon):
    df_train, df_valid, df_test = _split_dataset(df)
    mean, std = df_train.mean().values, df_train.std().values
    data_train, data_valid, data_test = (_df_to_io(df, history, horizon, mean, std)
                                         for df in (df_train, df_valid, df_test))

    return data_train, data_valid, data_test, mean, std


def prepare_case(df, history, horizon):
    df_train, df_valid, df_test = _split_dataset(df)
    mean, std = df_train.mean(), df_train.std()
    df_week = df[df.index.weekofyear == (df.index.weekofyear[0] + 1)]
    df_days = [df_week[df_week.index.weekday == day] for day in range(7)]
    data = (_df_to_io(df, history, horizon, mean, std) for df in df_days)
    return data, mean, std


def _split_dataset(df, train_ratio=0.7, test_ratio=0.2):

    def _get_days(index):
        date_s, date_e = index[0], index[-1]
        days = (date_e - date_s).days + 1
        assert (len(index) % days) == 0
        return days

    # calculate dates
    days = _get_days(df.index)
    days_train = pd.Timedelta(days=round(days * train_ratio))
    days_test = pd.Timedelta(days=round(days * test_ratio))
    date_train = df.index[0].date() + days_train
    date_test = df.index[-1].date() - days_test
    # select df
    dateindex = df.index.date
    df_train = df[dateindex < date_train]
    df_valid = df[(dateindex >= date_train) & (dateindex < date_test)]
    df_test = df[dateindex >= date_test]
    return df_train, df_valid, df_test


def _df_to_io(df, history, horizon, mean, std):
    def _gen_seq(arr, length):
        samples = arr.shape[0] - length + 1
        seq = np.stack([arr[i:i + length] for i in range(samples)], axis=0)
        return seq

    # data, time, weekday
    data = df.fillna(method='ffill').fillna(method='bfill').values
    data = (data - mean) / std
    _, time = np.unique(df.index.time, return_inverse=True)
    weekday = df.index.weekday
    # input sequences
    data_len = history + horizon - 1
    data, time, weekday = (_gen_seq(dat[:-1], data_len)
                           for dat in (data, time, weekday))
    # output sequences
    targets = df.values[history:]
    targets = _gen_seq(targets, horizon)
    aeq(len(data), len(time), len(weekday), len(targets))
    return data, time, weekday, targets


class SparseTimeSeries:
    def __init__(self, ts, history, horizon):
        self.history = history
        self.horizon = horizon
        ss_train, ss_valid, ss_test = _split_dataset(ts)
        self.data_train = self._gen_seq_coo(ss_train)
        self.data_valid = self._gen_seq_coo(ss_valid)
        self.data_test = self._gen_seq_coo(ss_test)

    def _gen_seq_coo(self, ts):
        coos = self._ss_to_coo(ts)
        days = _get_days(ts.index.levels[0])
        times = len(coos) // days
        samples = times - self.horizon - self.history + 1
        ret = []
        for day in range(days):
            anchor = day * times
            ret.extend([coos[anchor + i:anchor + i + self.history]
                        for i in range(samples)])
        return ret

    @staticmethod
    def _ss_to_coo(ts):
        coos = []
        for datetime in ts.index.levels[0]:
            val = ts[datetime].values.tolist()
            row = ts[datetime].index.get_level_values(0).tolist()
            col = ts[datetime].index.get_level_values(1).tolist()
            coos.append(((row, col), val))
        return coos
