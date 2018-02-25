import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from Data import TrafficDataset


class Dataset3(Dataset):
    def __init__(self, data_numerical, data_categorical, targets):
        self.data_numerical = data_numerical
        self.data_categorical = data_categorical
        self.targets = targets

    def __getitem__(self, index):
        return self.data_numerical[index], self.data_categorical[index], self.targets[index]

    def __len__(self):
        return self.data_numerical.size(0)


def getDataset(dataset='highway', freq=15, start=360, past=120, future=60):

    def getTrainValidTest(data):
        assert data.dim() is 4
        days_valid = data.size(0) // 6
        days_train = data.size(0) - 2 * days_valid
        data_train = data[:days_train]
        data_valid = data[days_train:-days_valid]
        data_test = data[-days_valid:]
        return data_train, data_valid, data_test

    dataset = TrafficDataset(dataset, freq=freq, start=start, past=past, future=future)
    data_numerical = getTrainValidTest(torch.FloatTensor(dataset.data_numerical))
    data_categorical = getTrainValidTest(orch.LongTensor(dataset.data_categorical))
    targets = getTrainValidTest(torch.FloatTensor(dataset.targets))

    data_train, data_valid, data_test = (
        Dataset3(data_numerical=data_numerical[i],
                 data_categorical=data_categorical[i],
                 targets=targets[i])
        for i in range(3)
    )

    data_train = DataLoader(
        dataest=data_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    data_valid = DataLoader(
        dataest=data_valid,
        batch_size=args.batch_size,
        pin_memory=True
    )

    data_test = DataLoader(
        dataest=data_test,
        batch_size=args.batch_size,
        pin_memory=True
    )

    mean, std = torch.FloatTensor(dataset.mean), torch.FloatTensor(dataset.std)

    return data_train, data_valid, data_test, mean, std


def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def pad_head(dim, head):
    pad = head - dim % head
    dim += pad
    return dim, (pad // 2, pad // 2 + pad % 2)

def get_mask_trim(length, past):
    shape = (length, length)
    mask_past = np.tril(np.ones(shape), k=-past).astype('uint8')
    mask_future = np.triu(np.ones(shape), k=1).astype('uint8')
    mask_past = torch.from_numpy(mask_past)
    mask_future = torch.from_numpy(mask_future)
    return mask_future + mask_past


def get_mask_dilated(length, dilations):
    def _get_mask_dilated(length, dilation, window):
        attn_shape = (length, length)
        mask = np.ones(attn_shape)
        for i in range(window):
            k = -i * dilation
            mask -= np.diag(np.ones(length + k), k)
        mask = torch.from_numpy(mask.astype('uint8'))
        return mask

    masks = []
    for i in range(len(dilations) - 1):
        dilation = dilations[i]
        window = dilations[i + 1] // dilation
        mask = _get_mask_dilated(length, dilation, window)
        masks.append(mask)
    return torch.stack(masks, 0)


def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.cpu().numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
