import time
import math

import torch
from torch.autograd import Variable

import Data
import Consts


def split_data(data, unit):
    data_train = data[:unit * Consts.DAYS_TRAIN]
    data_valid = data[unit * Consts.DAYS_TRAIN:unit * -Consts.DAYS_TEST]
    data_test = data[unit * -Consts.DAYS_TEST:]
    return data_train, data_valid, data_test


def load_data(args):
    if args.data_type == 'seq':
        flow, days, times, flow_mean, flow_std = Data.load_flow(
            gran=args.gran)
    else:
        flow, targets, days, times, flow_mean, flow_std = Data.load_flow(
            gran=args.gran, past=args.past, future=args.future)
        targets = torch.FloatTensor(targets)
        targets = targets.cuda() if args.gpuid else targets
    flow = Variable(torch.FloatTensor(flow))
    days = Variable(torch.LongTensor(days))
    times = Variable(torch.LongTensor(times))
    flow_mean = Variable(torch.FloatTensor(flow_mean))
    flow_std = Variable(torch.FloatTensor(flow_std))
    if args.gpuid:
        flow = flow.cuda()
        days = days.cuda()
        times = times.cuda()
        flow_mean = flow_mean.cuda()
        flow_std = flow_std.cuda()
    unit = flow.size(0) // Consts.DAYS
    flow_train, flow_valid, flow_test = split_data(flow, unit)
    days_train, days_valid, days_test = split_data(days, unit)
    times_train, times_valid, times_test = split_data(times, unit)
    if args.data_type == 'seq':
        return (flow_train, flow_valid, flow_test,
                days_train, days_valid, days_test,
                times_train, times_valid, times_test,
                flow_mean, flow_std)
    else:
        targets_train, targets_valid, targets_test = split_data(targets, unit)
        return (flow_train, flow_valid, flow_test,
                targets_train, targets_valid, targets_test,
                days_train, days_valid, days_test,
                times_train, times_valid, times_test,
                flow_mean, flow_std)


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
