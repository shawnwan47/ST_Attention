import datetime
import numpy as np
import pandas as pd

from constants import EPS
from lib.utils import aeq


def prepare_dataset(df, start, end, bday, history, horizon):
    df = _filter_df(df, start, end, bday)
    df_train, df_valid, df_test = _split_dataset(df)
    mean, std = df_train.mean().values, df_train.std().values
    data_train, data_valid, data_test = (_df_to_io(df, history, horizon, mean, std)
                                         for df in (df_train, df_valid, df_test))
    return data_train, data_valid, data_test, mean, std


def _filter_df(df, start, end, bday):
    time_filter = (df.index.hour >= start) & (df.index.hour < end)
    bday_filter = df.index.weekday < 5
    if bday:
        return df[time_filter & bday_filter]
    else:
        return df[time_filter]


def prepare_case(df, history, horizon):
    df_train, df_valid, df_test = _split_dataset(df)
    mean, std = df_train.mean(), df_train.std()
    df_week = df[df.index.weekofyear == (df.index.weekofyear[0] + 1)]
    df_days = [df_week[df_week.index.weekday == day] for day in range(7)]
    data = (_df_to_io(df, history, horizon, mean, std) for df in df_days)
    return data, mean, std


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


def _gen_seq(arr, num_samples, length):
    assert num_samples < arr.shape[0] - length
    seq = np.stack([arr[i:i + length] for i in range(num_samples)], axis=0)
    return seq


def _gen_daily_seqs(arr, samples, length):
    assert samples < arr.shape[1] - length
    seqs = np.stack([arr[:, i:i + length] for i in range(samples)], axis=1)
    if len(seqs.shape) is 3:
        return seqs.reshape(-1, length)
    elif len(seqs.shape) is 4:
        return seqs.reshape(-1, length, arr.shape[-1])
    else:
        raise(Exception('seqs not right.'))


def _df_to_io(df, history, horizon, mean, std, framework, model):
    # data
    data = (df.values - mean) / std
    data[np.isnan(data)] = 0
    targets = df.values
    _, time = np.unique(df.index.time, return_inverse=True)
    weekday = df.index.weekday

    # split days
    days = _get_days(df)
    dim = df.shape[1]

    data = data.reshape(days, -1, dim)
    targets = targets.reshape(days, -1, dim)[:, history:]
    time = time.reshape(days, -1)
    weekday = time.reshape(days, -1)

    # samples & length
    daily_samples = data.shape[1] - horizon
    length = history + horizon - 1 if framework is 'seq2seq' else history

    targets = _gen_daily_seqs(targets, daily_samples, horizon).transpose(0, 2, 1)
    data = _gen_daily_seqs(data, daily_samples, length)
    if framework is 'vec2vec':
        time = time[:, :daily_samples].reshape(-1)
        weekday = weekday[:, :daily_samples].reshape(-1)
    else:
        time = _gen_daily_seqs(time, daily_samples, length)
        weekday = _gen_daily_seqs(weekday, daily_samples, length)

    # model io
    if model is not 'RNN':
        time = np.repeat(time, dim).reshape(-1, dim)
        weekday = np.repeat(weekday, dim).reshape(-1, dim)
    if model in ['MLP', 'GraphAttention', 'GraphAttentionFusion']:
        data = data.transpose(0, 2, 1)
    elif model in ['GraphRNN', 'GraphAttentionRNN', 'GraphAttentionFusionRNN']:
        data = np.expand_dimensions(data, -1)

    return data, time, weekday, targets
