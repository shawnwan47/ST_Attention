import numpy as np
import torch
from torch.autograd import Variable

import Data


def load_adj():
    return torch.ByteTensor(Data.load_adj()).cuda() == 0


def denormalize(flow, flow_mean, flow_std):
    return flow * flow_std + flow_mean


def load_data(args):
    flow, flow_min, flow_scale = Data.load_flow_pixel(args.flow_size)
    day, time = Data.load_daytime()
    loc = Data.load_loc()

    flow = torch.LongTensor(flow)
    day = torch.LongTensor(day)
    time = torch.LongTensor(time)
    loc = torch.LongTensor(loc)

    inp = torch.cat((flow, day, time, loc), -1)
    inp = torch.cat((inp[:-1], inp[1:]), 1)
    tgt = flow[1:]

    inp_train = Variable(inp[:args.days_train])
    inp_valid = Variable(inp[args.days_train:-args.days_test], volatile=True)
    inp_test = Variable(inp[-args.days_test:], volatile=True)
    tgt_train = Variable(tgt[:args.days_train])
    tgt_valid = Variable(tgt[args.days_train:-args.days_test], volatile=True)
    tgt_test = Variable(tgt[-args.days_test:], volatile=True)

    flow_min = Variable(torch.FloatTensor(flow_min), require_grad=False).cuda()
    flow_scale = Variable(torch.FloatTensor(flow_scale), require_grad=False).cuda()

    return (inp_train, inp_valid, inp_test,
            tgt_train, tgt_valid, tgt_test,
            flow_min, flow_scale)


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
