import numpy as np
import torch
from torch.autograd import Variable

import Data


def load_adj():
    return torch.ByteTensor(Data.load_adj()).cuda() == 0


def denormalize(flow, flow_mean, flow_std):
    return flow * flow_std + flow_mean


def load_data_highway(args):
    flow, diff, flow_mean, flow_std, daytime = Data.load_flow_highway()
    flow = torch.FloatTensor(flow).cuda()
    diff = torch.FloatTensor(diff).cuda()
    flow_mean = torch.FloatTensor(flow_mean).cuda()
    flow_std = torch.FloatTensor(flow_std).cuda()
    daytime = torch.LongTensor(daytime).cuda()

    flow = torch.cat([flow[i:args.days - args.past_days + i]
                      for i in range(args.past_days + 1)], 1)
    diff = torch.cat([diff[i:args.days - args.past_days + i]
                      for i in range(args.past_days + 1)], 1)
    daytime = torch.cat([daytime[i:args.days - args.past_days + i]
                         for i in range(args.past_days + 1)], 1)

    def split_dataset(data):
        data_train = data[:args.days_train]
        data_valid = data[args.days_train:-args.days_test]
        data_test = data[-args.days_test:]
        return data_train, data_valid, data_test

    def cat_tgt(tgt):
        return torch.stack([tgt[:, i:tgt.size(1) - args.future + i + 1]
                            for i in range(args.future)], -2)

    flow_train, flow_valid, flow_test = split_dataset(flow)
    diff_train, diff_valid, diff_test = split_dataset(diff)
    daytime_train, daytime_valid, daytime_test = split_dataset(daytime)

    tgt_train = flow_train[:, args.past + 1:]
    tgt_valid = flow_valid[:, args.past + 1:]
    tgt_test = flow_test[:, args.past + 1:]
    tgt_train = cat_tgt(denormalize(tgt_train, flow_mean, flow_std))
    tgt_valid = cat_tgt(denormalize(tgt_valid, flow_mean, flow_std))
    tgt_test = cat_tgt(denormalize(tgt_test, flow_mean, flow_std))

    flow_train = flow_train[:, :-args.future].contiguous()
    flow_valid = flow_valid[:, :-args.future].contiguous()
    flow_test = flow_test[:, :-args.future].contiguous()
    diff_train = diff_train[:, :-args.future].contiguous()
    diff_valid = diff_valid[:, :-args.future].contiguous()
    diff_test = diff_test[:, :-args.future].contiguous()
    daytime_train = daytime_train[:, :-args.future].contiguous()
    daytime_valid = daytime_valid[:, :-args.future].contiguous()
    daytime_test = daytime_test[:, :-args.future].contiguous()

    print('Data loaded.\n flow: {}, target: {}, time: {}'.format(
        flow_train.size(), tgt_train.size(), daytime_train.size()))
    return (flow_train, flow_valid, flow_test,
            diff_train, diff_valid, diff_test,
            tgt_train, tgt_valid, tgt_test,
            daytime_train, daytime_valid, daytime_test,
            flow_mean, flow_std)


def load_data_metro(args):
    data = Data.load_flow(args.resolution, args.start_time, args.end_time)
    flow, daytime, flow_mean, flow_std = data
    flow = torch.FloatTensor(flow).cuda()
    daytime = torch.LongTensor(daytime).cuda()
    flow_mean = torch.FloatTensor(flow_mean).cuda()
    flow_std = torch.FloatTensor(flow_std).cuda()

    slices = flow.size(0) // args.days

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
        idx_train = generate_idx(idx[:args.days_train], batch)
        idx_valid = generate_idx(idx[args.days_train:-args.days_test], batch)
        idx_test = generate_idx(idx[-args.days_test:], batch)
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


def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def get_mask_trim(length, past):
    attn_shape = (length, length)
    mask_past = np.tril(np.ones(attn_shape), k=-past).astype('uint8')
    mask_future = np.triu(np.ones(attn_shape), k=1).astype('uint8')
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
