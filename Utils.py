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
    days_valid = data.shape[0] // 6
    days_train = data.shape[0] - 2 * days_valid
    *sizes, num_loc, dim = data.shape
    new_size = (-1, num_loc, dim)
    data_train = Variable(data[:days_train].contiguous().view(new_size))
    data_valid = Variable(data[days_train:-days_valid].contiguous().view(new_size), volatile=True)
    data_test = Variable(data[-days_valid:].contiguous().view(new_size), volatile=True)
    return data_train, data_valid, data_test


def getDataset(dataset='highway', freq=15, start=360, past=120, future=60,
               inp='OD', out='OD', batch_size=128):
    dataset = Data.SpatialTraffic(dataset=dataset, freq=freq,
                                  start=start, past=past, future=future,
                                  inp=inp, out=out)

    data_num = getTrainValidTest(dataset.data_num)
    data_cat = getTrainValidTest(dataset.data_cat)
    targets = getTrainValidTest(dataset.targets)

    data_train = Dataset3(data_num=data_num[0], data_cat=data_cat[0], targets=targets[0])
    data_valid = Dataset3(data_num=data_num[1], data_cat=data_cat[1], targets=targets[1])
    data_test = Dataset3(data_num=data_num[2], data_cat=data_cat[2], targets=targets[2])

    data_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    data_valid = DataLoader(dataset=data_valid, batch_size=batch_size)
    data_test = DataLoader(dataset=data_test, batch_size=batch_size)

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
