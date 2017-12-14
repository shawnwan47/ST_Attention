import numpy as np
import torch
from torch.autograd import Variable

import Data
from Consts import DAYS, DAYS_TRAIN, DAYS_TEST, MAX_SEQ_LEN


def load_data_highway(args):
    flow, daytime, flow_mean, flow_std = Data.load_flow_highway()
    flow = torch.FloatTensor(flow).cuda()
    daytime = torch.LongTensor(daytime).cuda()
    flow_mean = torch.FloatTensor(flow_mean).cuda()
    flow_std = torch.FloatTensor(flow_std).cuda()

    if args.past_days:
        flow = torch.cat((flow[i:args.days - args.past_days + i]
                          for i in range(args.past_days)), 1)
        daytime = torch.cat((daytime[i:args.days - args.past_days + i]
                             for i in range(args.past_days)), 1)

    def split_dataset(data):
        data_train = data[:args.days_train]
        data_valid = data[args.days_train:-args.days_test]
        data_test = data[-args.days_test:]
        return data_train, data_valid, data_test

    def cat_tgt(tgt):
        length = tgt.size(1)
        return torch.cat([tgt[:, i:length - args.future + i + 1]
                          for i in range(future)], -1)

    flow_train, flow_valid, flow_test = split_dataset(flow)
    daytime_train, daytime_valid, daytime_test = split_dataset(daytime)

    inp_train = flow_train[:, :-args.future]
    inp_valid = flow_valid[:, :-args.future]
    inp_test = flow_test[:, :-args.future]
    daytime_train = daytime_train[:, :-args.future]
    daytime_valid = daytime_valid[:, :-args.future]
    daytime_test = daytime_test[:, :-args.future]
    tgt_train = flow_train[:, args.past:]
    tgt_valid = flow_valid[:, args.past:]
    tgt_test = flow_test[:, args.past:]
    tgt_train = denormalize(tgt_train, flow_mean, flow_std)
    tgt_valid = denormalize(tgt_valid, flow_mean, flow_std)
    tgt_test = denormalize(tgt_test, flow_mean, flow_std)
    tgt_train = cat_tgt(tgt_train)
    tgt_valid = cat_tgt(tgt_valid)
    tgt_test = cat_tgt(tgt_test)

    print('Data loaded.\n flow: {}, target: {}, time: {}'.format(
        inp_train.size(), tgt_train.size(), daytime_train.size()))
    return (inp_train, inp_valid, inp_test,
            tgt_train, tgt_valid, tgt_test,
            daytime_train, daytime_valid, daytime_test,
            flow_mean, flow_std)


def load_data_metro(args):
    data = Data.load_flow(args.gran, args.start_time, args.end_time)
    flow, daytime, flow_mean, flow_std = data
    flow = torch.FloatTensor(flow).cuda()
    daytime = torch.LongTensor(daytime).cuda()
    flow_mean = torch.FloatTensor(flow_mean).cuda()
    flow_std = torch.FloatTensor(flow_std).cuda()

    slices = flow.size(0) // DAYS

    def init_idx(batch):
        return torch.arange(batch).long().cuda()

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


def denormalize(flow, flow_mean, flow_std):
    return flow * flow_std + flow_mean


