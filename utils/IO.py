import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimeSeries:
    def __init__(self, df):
        scaler = StandardScaler().fit(df)
        self.index, self.values = df.index, scaler.transform(df)
        self.mean, self.scale = scaler.mean_, scaler.scale_

    def gen_daily_seq_io(self, start=6, end=23):
        # select time range
        days = (self.index[-1].date() - self.index[0].date()).days + 1
        indices = (self.index.hour >= start) & (self.index.hour < end)
        datetime, ts = self.index[indices], self.values[indices]
        # get day, time
        day = datetime.weekday
        _, time = np.unique(datetime.time, return_inverse=True)
        daytime = np.concatenate((day, time), -1)
        # gather daily sequence
        ts = ts.reshape(days, -1, ts.shape[1])
        daytime = daytime.reshape(days, -1, 2)
        # split data targets
        data_num, data_cat, targets = ts[:-1], daytime[:-1], ts[1:]
        return data_num, data_cat, targets

    @staticmethod
    def train_val_test_split(arr, val_ratio=0.1, test_ratio=0.2):
        n_sample = arr.shape[0]
        n_val = int(round(n_sample * val_ratio))
        n_test = int(round(n_sample * test_ratio))
        n_train = n_sample - n_val - n_test
        return arr[:n_train], arr[n_train:-n_test], arr[-n_test:]
