import torch
from torch.autograd import Variable

import Data
from Consts import DAYS, DAYS_TRAIN, DAYS_TEST, MODEL_PATH


def split_data(data, dim):
    assert dim in [0, 1]
    oneday = data.size(dim) // DAYS
    if dim == 0:
        data_train = data[:oneday * DAYS_TRAIN]
        data_valid = data[oneday * DAYS_TRAIN:oneday * -DAYS_TEST]
        data_test = data[oneday * -DAYS_TEST:]
    elif dim == 1:
        data_train = data[:, :oneday * DAYS_TRAIN]
        data_valid = data[:, oneday * DAYS_TRAIN:oneday * -DAYS_TEST]
        data_test = data[:, oneday * -DAYS_TEST:]
    return data_train, data_valid, data_test


def load_data(args):
    data = getattr(Data, 'load_flow_' + args.data_type)(args)
    inputs, targets, daytimes, flow_mean, flow_std = data
    inputs = torch.FloatTensor(inputs)
    targets = torch.FloatTensor(targets)
    daytimes = torch.LongTensor(daytimes)
    flow_mean = torch.FloatTensor(flow_mean)
    flow_std = torch.FloatTensor(flow_std)
    if args.gpuid:
        inputs = inputs.cuda()
        targets = targets.cuda()
        daytimes = daytimes.cuda()
        flow_mean = flow_mean.cuda()
        flow_std = flow_std.cuda()
    dim = 0
    if args.data_type == 'seq':
        dim = 1
        inputs = inputs.transpose(0, 1)
        targets = targets.transpose(0, 1)
        daytimes = daytimes.transpose(0, 1)
    inputs_train, inputs_valid, inputs_test = split_data(inputs, dim)
    targets_train, targets_valid, targets_test = split_data(targets, dim)
    daytimes_train, daytimes_valid, daytimes_test = split_data(daytimes, dim)
    return (inputs_train, inputs_valid, inputs_test,
            targets_train, targets_valid, targets_test,
            daytimes_train, daytimes_valid, daytimes_test,
            flow_mean, flow_std)


def modelpath(args):
    path = MODEL_PATH
    path += 'atn_' + args.attention_type if args.attention else ''
    path += 'hid' + str(args.nhid)
    path += 'lay' + str(args.nlay)
    return path


def tensor2VarRNN(t):
    return Variable(t.transpose(0, 1))


def var2np(x):
    return x.cpu().data.numpy()


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)
