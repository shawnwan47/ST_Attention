import numpy as np
import torch
from torch.autograd import Variable

import Data
from Consts import DAYS, DAYS_TRAIN, DAYS_TEST, MAX_SEQ_LEN


def load_data(args):
    data = Data.load_flow(args.gran, args.start_time, args.end_time)
    flow, daytime, flow_mean, flow_std = data
    flow = torch.FloatTensor(flow)
    daytime = torch.LongTensor(daytime)
    flow_mean = torch.FloatTensor(flow_mean)
    flow_std = torch.FloatTensor(flow_std)
    if args.gpuid:
        flow = flow.cuda()
        daytime = daytime.cuda()
        flow_mean = flow_mean.cuda()
        flow_std = flow_std.cuda()

    slices = flow.size(0) // DAYS

    def init_idx(batch):
        idx = torch.arange(batch).long()
        return idx.cuda() if args.gpuid else idx

    def generate_idx(idx, batch):
        ret = [idx + i * batch for i in range(slices)]
        return torch.cat(ret)

    def extend_yesterday(data):
        batch = data.size(0) // slices
        idx = init_idx(batch)
        idx_yesterday = generate_idx(idx[:-1], batch)
        idx_tomorrow = generate_idx(idx[1:], batch)
        return torch.cat((data[idx_yesterday], data[idx_tomorrow]), 1)

    def split_data(data):
        batch = data.size(0) // slices
        idx = init_idx(batch)
        idx_train = generate_idx(idx[:DAYS_TRAIN], batch)
        idx_valid = generate_idx(idx[DAYS_TRAIN:-DAYS_TEST], batch)
        idx_test = generate_idx(idx[-DAYS_TEST:], batch)
        return data[idx_train], data[idx_valid], data[idx_test]

    if args.yesterday:
        flow = extend_yesterday(flow)
        daytime = extend_yesterday(daytime)
    flow_train, flow_valid, flow_test = split_data(flow)
    daytime_train, daytime_valid, daytime_test = split_data(daytime)

    flow_train = flow_train.transpose(0, 1)
    flow_valid = flow_valid.transpose(0, 1)
    flow_test = flow_test.transpose(0, 1)
    daytime_train = daytime_train.transpose(0, 1)
    daytime_valid = daytime_valid.transpose(0, 1)
    daytime_test = daytime_test.transpose(0, 1)

    print('Data loaded.\ntrain: {}, valid: {}, test: {}'.format(
        flow_train.size(), flow_valid.size(), flow_test.size()))
    return (flow_train, flow_valid, flow_test,
            daytime_train, daytime_valid, daytime_test,
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


def _get_attn_subsequent_mask():
    attn_shape = (1, MAX_SEQ_LEN, MAX_SEQ_LEN)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask
