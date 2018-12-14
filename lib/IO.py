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
    if bday:
        bday_filter = df.index.weekday < 5
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
    days = len(np.unique(df.index.date))
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
    # extract data
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

    # generate samples
    assert framework in ['vec2vec', 'seq2vec', 'seq2seq']

    daily_samples = data.shape[1] - history - horizon
    num_samples = daily_samples * num_days
    length = history + horizon - 1 if framework == 'seq2seq' else history

    data = _gen_daily_seqs(data, daily_samples, length)
    targets = _gen_daily_seqs(targets, daily_samples, horizon)
    if framework == 'vec2vec':
        time = time[:, :daily_samples].reshape(-1)
        weekday = weekday[:, :daily_samples].reshape(-1)
    else:
        time = _gen_daily_seqs(time, daily_samples, length)
        weekday = _gen_daily_seqs(weekday, daily_samples, length)
    aeq(data.shape[0], targets.shape[0], time.shape[0], weekday.shape[0])

    # paradigm: io structure
    assert paradigm in ['temporal', 'spatial', 'st']
    if paradigm == 'temporal':
        pass
    else:
        time = np.expand_dims(time, -1)
        weekday = np.expand_dims(weekday, -1)
        if paradigm == 'spatial':
            data = data.transpose(0, 2, 1)
            targets = targets.transpose(0, 2, 1)
        elif paradigm == 'st':
            data = np.expand_dims(data, -1)
            if framework is 'seq2vec':
                targets = targets.transpose(0, 2, 1)
            else:
                targets = np.expand_dims(targets, -1)

    return data, time, weekday, targets


def _gen_daily_seqs(arr, samples, length):
    assert samples <= arr.shape[1] - length
    seqs = np.stack([arr[:, i:i + length] for i in range(samples)], axis=1)
    if len(seqs.shape) is 3:
        return seqs.reshape(-1, length)
    elif len(seqs.shape) is 4:
        return seqs.reshape(-1, length, arr.shape[-1])
    else:
        raise(Exception('seqs invalid.'))
