import torch
from torch.autograd import Variable

import Data
from Consts import DAYS, DAYS_TRAIN, DAYS_TEST


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
    if args.data_type == 'seq':
        inputs, targets, daytimes, flow_mean, flow_std = Data.load_flow_seq(
            gran=args.gran)
    else:
        inputs, targets, daytimes, flow_mean, flow_std = Data.load_flow_img(
            gran=args.gran, past=args.past, future=args.future)
    inputs = Variable(torch.FloatTensor(inputs))
    targets = Variable(torch.FloatTensor(targets))
    daytimes = Variable(torch.LongTensor(daytimes))
    flow_mean = Variable(torch.FloatTensor(flow_mean))
    flow_std = Variable(torch.FloatTensor(flow_std))
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


def modelname(args):
    name = args.description
    name += 'atn' + args.attn_type if args.attn else ''
    name += 'hid' + args.nhid
    name += 'lay' + args.nlay
    return name


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
