import datetime
import numpy as np
import pandas as pd


def get_rush_hours_bool_index(df, hours=((7, 10), (17, 20)), weekdays=(0, 5)):
    weekday_predate = (df.index.dayofweek >= weekdays[0]) & (df.index.dayofweek < weekdays[1])
    hour_predate = (df.index.time >= datetime.time(hours[0][0], 0)) & (df.index.time < datetime.time(hours[0][1], 0))
    hour_predate |= (df.index.time >= datetime.time(hours[1][0], 0)) & (df.index.time < datetime.time(hours[1][1], 0))
    return weekday_predate & hour_predate


def generate_graph_seq2seq_io_data_with_time(df, batch_size, seq_len, horizon, num_nodes, scaler=None,
                                             add_time_in_day=True, add_day_in_week=False):
    if scaler:
        df = scaler.transform(df)
    num_samples, _ = df.shape
    data = df.values
    batch_len = num_samples // batch_size
    data = np.expand_dims(data, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    data = data[:batch_size * batch_len, :, :].reshape((batch_size, batch_len, num_nodes, -1))
    epoch_size = batch_len - seq_len - horizon + 1
    x, y = [], []
    for i in range(epoch_size):
        x_i = data[:, i: i + seq_len, ...]
        y_i = data[:, i + seq_len: i + seq_len + horizon, :, :]
        x.append(x_i)
        y.append(y_i)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def train_val_test_split_df(df, val_ratio=0.1, test_ratio=0.2):
    n_sample, _ = df.shape
    n_val = int(round(n_sample * val_ratio))
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_val - n_test
    train_data, val_data, test_data = df.iloc[:n_train, :], df.iloc[n_train: n_train + n_val, :], df.iloc[-n_test:, :]
    return train_data, val_data, test_data
