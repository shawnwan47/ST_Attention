import os
import datetime

import pandas as pd
import numpy as np

import torch

from consts import *


# data loading
def load_data(affix):
    filepath = DATA_PATH + affix + '.csv'
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        data_files = sorted(os.listdir(DATA_PATH))
        data_files = list(filter(
            lambda x: x.startswith(affix + '_'), data_files))
        data = pd.concat([
            pd.read_csv(DATA_PATH + x, index_col=0, parse_dates=True)
            for x in data_files])
        data.to_csv(filepath, index=True)
    return data


def resample_data(data, freq):
    data = data.resample(str(freq) + 'T', 'sum')
    idx = data.index.time
    return data[(datetime.time(6, 0) <= idx) & (idx < datetime.time(22, 0))]


# data for model
def split_dataset(data):
    unit = data.size(0) // DAYS
    data_train = data[:unit * DAYS_TRAIN]
    data_valid = data[unit * DAYS_TRAIN:unit * -DAYS_TEST]
    data_test = data[unit * -DAYS_TEST:]
    return data_train, data_valid, data_test


def load_flow(freq, norm=False):
    flow_o = load_data('O')
    flow_d = load_data('D')
    flow = pd.concat((flow_o, flow_d), axis=1)
    flow = resample_data(flow, freq)
    mean, std = flow.mean(), flow.std()
    if norm:
        flow = (flow - mean) / std
        flow[flow != flow] = 0
    # flow.dropna(axis=1, inplace=True)
    # mean, std = mean[mean != 0], std[std != 0]
    flow = torch.FloatTensor(flow.as_matrix().tolist())
    flow = flow.view(DAYS, -1, flow.size(-1))
    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
    return flow, mean, std


def load_flow_seqs(freq):
    flow, mean, std = load_flow(freq, True)
    seqs = flow[:, :-1]
    vals = flow[:, 1:]
    return split_dataset(seqs), split_dataset(vals), mean, std


def load_flow_images(freq, nprev):
    flow, mean, std = load_flow(freq)
    flows = [flow[:, i:-nprev + i] for i in range(nprev)]
    images = torch.cat(flows, -1)
    images = images.view(-1, 1, nprev, flow.size(-1))
    labels = flow[:, nprev:]  # predict flow
    labels = labels.contiguous().view(images.size(0), 1, -1)
    return split_dataset(images), split_dataset(labels), mean, std


def load_od(freq, affix='od'):
    od = resample_data(load_data(affix.upper()), freq)
    mean, std = od.mean(), od.std()
    od = (od - mean) / std
    return od, mean, std


def load_daystime(freq, nprev=1):
    data = load_data('O')
    data = resample_data(data, freq)
    days = np.array(data.index.dayofweek).reshape(DAYS, -1)[:, nprev - 1:-1]
    time = np.arange(data.shape[0]).reshape(DAYS, -1)[:, nprev - 1:-1]
    time %= data.shape[0] // 22
    time -= np.min(time)
    days = torch.LongTensor(days.tolist())
    time = torch.LongTensor(time.tolist())
    if nprev > 1:
        days = days.view(-1, 1)
        time = time.view(-1, 1)
    return split_dataset(days), split_dataset(time)


def load_adj_spatial():
    return np.genfromtxt(DATA_PATH + 'link.txt')


def load_adj_od(od_name=DATA_PATH + 'od_al.csv'):
    ret = pd.read_csv(od_name, index_col=0)
    ret.columns = list(map(int, ret.columns))
    ret.index.name = ''
    return ret


if __name__ == '__main__':
    # flow_o = load_data('O')
    # flow_d = load_data('D')
    od = load_data('OD')
    do = load_data('DO')
    # flow = load_flow(15)
    # seqs, vals = load_flow_seqs(15)
    # images, labels, mean, std = load_flow_images(15, 8)
    # days_img, time_img = load_daystime(15, 4)
