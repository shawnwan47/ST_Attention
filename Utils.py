import numpy as np
import torch
from torch.autograd import Variable

import Data


def load_adj():
    return torch.ByteTensor(Data.load_adj()).cuda() == 0


def denormalize(flow, flow_mean, flow_std):
    return flow * flow_std + flow_mean


def load_data(args):
    flow, flow_min, flow_scale = Data.load_flow_pixel(args.num_flow)
    day, time = Data.load_daytime()
    loc = Data.load_loc()

    flow = torch.LongTensor(flow)
    day = torch.LongTensor(day).expand_as(flow)
    time = torch.LongTensor(time).expand_as(flow)
    loc = torch.LongTensor(loc)
    inp = tgt = flow.contiguous().view(-1, args.num_loc)
    st = torch.stack((day, time, loc), -1).view(-1, args.num_loc, 3)

    num_sample = (args.days - 1) * args.num_time
    inp_size = (args.days - 1, args.num_time, args.past, args.num_loc)
    tgt_size = (args.days - 1, args.num_time, args.future, args.num_loc)
    st_size = (args.days - 1, args.num_time, args.past, args.num_loc, 3)
    inp = torch.stack([inp[i:i + args.past]
                       for i in range(num_sample)], 0).view(inp_size)
    tgt = torch.stack([tgt[i + args.past:i + args.past + args.future]
                       for i in range(num_sample)], 0).view(tgt_size)
    st = torch.stack([st[i:i + args.past]
                      for i in range(num_sample)], 0).view(st_size)

    inp_size = (-1, args.past, args.num_loc)
    tgt_size = (-1, args.future, args.num_loc)
    st_size = (-1, args.past, args.num_loc, 3)

    inp_train = Variable(inp[:args.days_train]).view(inp_size)
    inp_valid = Variable(inp[args.days_train:-args.days_test], volatile=True).view(inp_size)
    inp_test = Variable(inp[-args.days_test:], volatile=True).view(inp_size)
    tgt_train = Variable(tgt[:args.days_train]).view(tgt_size)
    tgt_valid = Variable(tgt[args.days_train:-args.days_test], volatile=True).view(tgt_size)
    tgt_test = Variable(tgt[-args.days_test:], volatile=True).view(tgt_size)
    st_train = Variable(st[:args.days_train]).view(st_size)
    st_valid = Variable(st[args.days_train:-args.days_test], volatile=True).view(st_size)
    st_test = Variable(st[-args.days_test:], volatile=True).view(st_size)

    flow_min = Variable(torch.FloatTensor(flow_min), requires_grad =False).cuda()
    flow_scale = Variable(torch.FloatTensor(flow_scale), requires_grad =False).cuda()

    return (inp_train, inp_valid, inp_test,
            tgt_train, tgt_valid, tgt_test,
            st_train, st_valid, st_test,
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


def get_mask_pixel(dim, past, future):
    shape = (future * dim, (past + future) * dim)



def torch2npsave(filename, data):
    def _var2np(x):
        return x.data.cpu().numpy()

    if type(data) in [tuple, list]:
        for i, d in enumerate(data):
            torch2npsave(filename + '_' + str(i), d)
    else:
        np.save(filename, _var2np(data))
