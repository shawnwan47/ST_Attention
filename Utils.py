import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from Data import TrafficData


class Dataset3(Dataset):
    def __init__(self, data_numerical, data_categorical, targets):
        self.data_numerical = data_numerical
        self.data_categorical = data_categorical
        self.targets = targets

    def __getitem__(self, index):
        return self.data_numerical[index], self.data_categorical[index], self.targets[index]

    def __len__(self):
        return self.data_numerical.size(0)


def getOD(data, od):
    stations = data.size(-2) // 2
    resize = False
    if data.dim() is 4:
        num_day, num_time, num_loc, size_last = data.size()
        data = data.view(-1, num_loc, size_last)
        resize = True
    if od == 'O':
        data = data[:, :stations].contiguous()
    elif od == 'D':
        data = data[:, -stations:].contiguous()
    if resize:
        data = data.view(num_day, num_time, -1, size_last)
    return data


def getDataset(dataset='highway', freq=15, start=360, past=120, future=60,
               out='D', batch_size=100):

    def getTrainValidTest(data):
        assert data.dim() is 4
        days_valid = data.size(0) // 6
        days_train = data.size(0) - 2 * days_valid
        *sizes, num_loc, dim = data.size()
        new_size = (-1, num_loc, dim)
        data_train = data[:days_train].contiguous().view(new_size)
        data_valid = data[days_train:-days_valid].contiguous().view(new_size)
        data_test = data[-days_valid:].contiguous().view(new_size)
        return data_train, data_valid, data_test

    dataset = TrafficData(dataset, freq=freq, start=start, past=past, future=future)
    data_numerical = torch.FloatTensor(dataset.data_numerical)
    data_categorical = torch.LongTensor(dataset.data_categorical)
    targets = torch.FloatTensor(dataset.targets).contiguous()
    targets = getOD(targets, out)

    data_numerical = getTrainValidTest(data_numerical)
    data_categorical = getTrainValidTest(data_categorical)
    targets = getTrainValidTest(targets)

    data_train = Dataset3(data_numerical=data_numerical[0],
                          data_categorical=data_categorical[0],
                          targets=targets[0])

    data_valid = Dataset3(data_numerical=Variable(data_numerical[1], volatile=True),
                          data_categorical=Variable(data_categorical[1], volatile=True),
                          targets=Variable(targets[1], volatile=True))

    data_test = Dataset3(data_numerical=Variable(data_numerical[2], volatile=True),
                         data_categorical=Variable(data_categorical[2], volatile=True),
                         targets=Variable(targets[2], volatile=True))

    data_train = DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    data_valid = DataLoader(
        dataset=data_valid,
        batch_size=batch_size,
        pin_memory=True
    )

    data_test = DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        pin_memory=True
    )

    mean = Variable(torch.FloatTensor(dataset.mean)).cuda()
    std = Variable(torch.FloatTensor(dataset.std)).cuda()
    size = (1, mean.size(0), 1)
    mean, std = getOD(mean.view(size), out), getOD(std.view(size), out)

    return data_train, data_valid, data_test, mean, std


def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.cpu().numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
