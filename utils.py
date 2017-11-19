import os

import pandas as pd
import numpy as np

from consts import *


def load_adj_spatial():
    return np.genfromtxt(DATA_PATH + 'link.txt')


def load_adj_od(od_name=DATA_PATH + 'od_al.csv'):
    ret = pd.read_csv(od_name, index_col=0)
    ret.columns = list(map(int, ret.columns))
    ret.index.name = ''
    return ret


def load_flow_data(affix, freq=15):
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
    data.columns = list(map(int, data.columns))

    return data.resample(str(freq) + 'T').sum()


# data for model
def load_flow(freq):
    flow_o = load_flow_data('O', freq)
    flow_d = load_flow_data('D', freq)
    flow = pd.concat((flow_o, flow_d), axis=1)
    mean, std = flow.mean(), flow.std()
    flow = (flow - mean) / std
    flow.fillna(0, inplace=True)
    # flow[flow != flow] = 0
    # flow.dropna(axis=1, inplace=True)
    # mean, std = mean[mean != 0], std[std != 0]
    # flow = torch.FloatTensor(flow.as_matrix().tolist())
    # flow = flow.view(DAYS, -1, flow.size(-1))
    # mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
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


def load_daystime(freq):
    data = load_flow_data('O', freq)
    days = np.array(data.index.dayofweek).reshape(DAYS, -1)
    time = np.arange(data.shape[0]).reshape(DAYS, -1)
    time %= data.shape[0] // DAYS
    return split_dataset(days), split_dataset(time)


def split_dataset(data):
    unit = data.shape[0] // DAYS
    data_train = data[:unit * DAYS_TRAIN]
    data_valid = data[unit * DAYS_TRAIN:unit * -DAYS_TEST]
    data_test = data[unit * -DAYS_TEST:]
    return data_train, data_valid, data_test


if __name__ == '__main__':
    flow, mean, std = load_flow(15)
    # flow_d = load_flow_data('D')
    # od = load_flow_data('OD')
    # do = load_flow_data('DO')
    # flow = load_flow(15)
    # seqs, vals = load_flow_seqs(15)
    # images, labels, mean, std = load_flow_images(15, 8)
    # days_img, time_img = load_daystime(15, 4)
    pass
