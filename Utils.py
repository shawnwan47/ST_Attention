import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import Data


class Dataset3(Dataset):
    def __init__(self, data_num, data_cat, targets):
        self.data_num = data_num
        self.data_cat = data_cat
        self.targets = targets

    def __getitem__(self, index):
        return self.data_num[index], self.data_cat[index], self.targets[index]

    def __len__(self):
        return self.targets.size(0)


def getTrainValidTest(data):
    days_valid = data.size(0) // 6
    days_train = data.size(0) - 2 * days_valid
    *sizes, num_loc, dim = data.size()
    new_size = (-1, num_loc, dim)
    data_train = data[:days_train].contiguous().view(new_size)
    data_valid = data[days_train:-days_valid].contiguous().view(new_size)
    data_test = data[-days_valid:].contiguous().view(new_size)
    return data_train, data_valid, data_test


def getDataset(dataset='highway', freq=15, start=360, past=120, future=60, batch_size=128):
    dataset = Data.SpatialTraffic(dataset=dataset, freq=freq,
                                  start=start, past=past, future=future)

    data_num = getTrainValidTest(torch.FloatTensor(dataset.data_num))
    data_cat = getTrainValidTest(torch.LongTensor(dataset.data_cat))
    targets = getTrainValidTest(torch.FloatTensor(dataset.targets))

    data = [Dataset3(data_num[i], data_cat[i], targets[i]) for i in range(3)]

    data_train = DataLoader(data[0], batch_size=batch_size, shuffle=True)
    data_valid = DataLoader(data[1], batch_size=batch_size)
    data_test = DataLoader(data[2], batch_size=batch_size)

    mean = Variable(torch.FloatTensor(dataset.mean)).unsqueeze(0).cuda()
    scale = Variable(torch.FloatTensor(dataset.scale)).unsqueeze(0).cuda()

    return data_train, data_valid, data_test, mean, scale


def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def get_mask_od(num_loc, od):
    pass


def get_mask_graph(dataset):
    loader = Data.Loader(dataset)
    dist = loader.load_dist()
    od = loader.load_od_sum()


def get_mask_target():
    pass

def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.cpu().numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
