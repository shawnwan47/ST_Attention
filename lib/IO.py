import datetime
import numpy as np
import pandas as pd

from constants import EPS
from lib.utils import aeq


def prepare_dataset(df, bday, start, end):
    df = _filter_df(df, bday, start, end)
    df_train, df_valid, df_test = _split_dataset(df)
    mean, std = df_train.mean().values, df_train.std().values
    data_train = _df_to_io((df_train - mean) / (std + EPS))
    data_valid = _df_to_io((df_valid - mean) / (std + EPS))
    data_test = _df_to_io((df_test - mean) / (std + EPS))
    return data_train, data_valid, data_test, mean, std


def _filter_df(df, bday, start, end):
    time_filter = (df.index.hour >= start) & (df.index.hour < end)
    if bday:
        bday_filter = df.index.weekday < 5
        return df[time_filter & bday_filter]
    else:
        return df[time_filter]


def _split_dataset(df, train_ratio=0.7, test_ratio=0.2):
    # calculate dates
    days = len(np.unique(df.index.date))
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
