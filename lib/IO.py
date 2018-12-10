import datetime
import numpy as np
import pandas as pd

from constants import EPS
from lib.utils import aeq


def prepare_dataset(df, bday, start, end, history, horizon, framework, paradigm):
    df = _filter_df(df, bday, start, end)
    df_train, df_valid, df_test = _split_dataset(df)
    mean, std = df_train.mean().values, df_train.std().values
    data_train = _df_to_io(df_train, history, horizon, mean, std, framework, paradigm)
    data_valid = _df_to_io(df_valid, history, horizon, mean, std, framework, paradigm)
    data_test = _df_to_io(df_test, history, horizon, mean, std, framework, paradigm)
    return data_train, data_valid, data_test, mean, std


def _filter_df(df, bday, start, end):
    time_filter = (df.index.hour >= start) & (df.index.hour < end)
    bday_filter = df.index.weekday < 5
    if bday:
        return df[time_filter & bday_filter]
    else:
        return df[time_filter]

#
# def prepare_case(df, history, horizon):
#     df_train, df_valid, df_test = _split_dataset(df)
#     mean, std = df_train.mean(), df_train.std()
#     df_week = df[df.index.weekofyear == (df.index.weekofyear[0] + 1)]
#     df_days = [df_week[df_week.index.weekday == day] for day in range(7)]
#     data = (_df_to_io(df, history, horizon, mean, std) for df in df_days)
#     return data, mean, std


def _get_days(df):
    date_s, date_e = df.index[0], df.index[-1]
    days = (date_e - date_s).days + 1
    assert (df.shape[0] % days) == 0
    return days


def _split_dataset(df, train_ratio=0.7, test_ratio=0.2):
    # calculate dates
    days = _get_days(df)
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


def _df_to_io(df, history, horizon, mean, std, framework, paradigm):
    # data
    data = (df.values - mean) / std
    data[np.isnan(data)] = 0
    targets = df.values
    _, time = np.unique(df.index.time, return_inverse=True)
    weekday = np.array(df.index.weekday)

    # split days
    num_days = _get_days(df)
    num_nodes = df.shape[1]

    data = data.reshape(num_days, -1, num_nodes)
    targets = targets.reshape(num_days, -1, num_nodes)[:, history:]
    time = time.reshape(num_days, -1)
    weekday = weekday.reshape(num_days, -1)

    # framework: samples & length & data shape
    daily_samples = data.shape[1] - history - horizon
    num_samples = daily_samples * num_days
    length = history + horizon - 1 if framework == 'seq2seq' else history

    data = _gen_daily_seqs(data, daily_samples, length)
    targets = _gen_daily_seqs(targets, daily_samples, horizon)
    if framework in ['none', 'vec2vec']:
        time = time[:, :daily_samples].reshape(-1)
        weekday = weekday[:, :daily_samples].reshape(-1)
    else:
        time = _gen_daily_seqs(time, daily_samples, length)
        weekday = _gen_daily_seqs(weekday, daily_samples, length)

    # paradigm: io structure
    if paradigm == 'none':
        data = data.reshape(num_samples, -1)
    elif paradigm == 'spatial':
        data = data.transpose(0, 2, 1)
        time = np.repeat(time, num_nodes).reshape(num_samples, num_nodes)
        weekday = np.repeat(weekday, num_nodes).reshape(num_samples, num_nodes)
    elif paradigm == 'spatialtemporal':
        data = np.expand_dims(data, -1)
        time = np.repeat(time, num_nodes, 1).reshape(num_samples, length, num_nodes)
        weekday = np.repeat(weekday, num_nodes, 1).reshape(num_samples, length, num_nodes)
    else:
        pass

    return data, time, weekday, targets


def _gen_daily_seqs(arr, samples, length):
    assert samples <= arr.shape[1] - length
    seqs = np.stack([arr[:, i:i + length] for i in range(samples)], axis=1)
    if len(seqs.shape) is 3:
        return seqs.reshape(-1, length)
    elif len(seqs.shape) is 4:
        return seqs.reshape(-1, length, arr.shape[-1])
    else:
        raise(Exception('seqs not right.'))
