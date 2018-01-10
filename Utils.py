import numpy as np
import torch
from torch.autograd import Variable

import Data


def load_adj():
    return torch.ByteTensor(Data.load_adj()).cuda() == 0


def load_data(args):
    orig, orig_min, orig_scale, dest, dest_min, dest_scale = Data.load_flow_pixel(args.num_flow)
    day, time = Data.load_daytime()
    loc_orig, loc_dest = Data.load_loc()

    orig = torch.LongTensor(orig)
    dest = torch.LongTensor(dest)
    day = torch.LongTensor(day).expand_as(orig)
    time = torch.LongTensor(time).expand_as(orig)
    loc_orig = torch.LongTensor(loc_orig)
    loc_dest = torch.LongTensor(loc_dest)

    inp = torch.stack((orig, day, time, loc_orig), -1)
    tgt = torch.stack((dest, day, time, loc_dest), -1)
    inp = inp.view(-1, inp.size(-2), inp.size(-1))
    tgt = tgt.view(-1, tgt.size(-2), tgt.size(-1))

    num_sample = (args.days - 1) * args.num_time
    inp = torch.stack([inp[i:i + args.past]
                       for i in range(num_sample)], 0)
    tgt = torch.stack([tgt[i + args.past:i + args.past + args.future]
                       for i in range(num_sample)], 0)

    train_size = args.days_train * args.num_time
    test_size = args.days_test * args.num_time

    inp_train = inp[:train_size]
    inp_valid = inp[train_size:-test_size]
    inp_test = inp[-test_size:]
    tgt_train = tgt[:train_size]
    tgt_valid = tgt[train_size:-test_size]
    tgt_test = tgt[-test_size:]

    tgt_min = Variable(torch.FloatTensor(dest_min), requires_grad=False).cuda()
    tgt_scale = Variable(torch.FloatTensor(dest_scale), requires_grad=False).cuda()

    print('Data loaded.\ninp: {}, tgt: {}'.format(
        inp_test.size(), tgt_test.size()))

    return (inp_train, inp_valid, inp_test,
            tgt_train, tgt_valid, tgt_test,
            tgt_min, tgt_scale)


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
