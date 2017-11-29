import random

import torch
from torch.autograd import Variable

from Constants import USE_CUDA


def encodeRNN(var_inp, encoder):
    bsz = var_inp.size(1)
    len_inp = var_inp.size(0)
    # encoding
    enc_hid = encoder.initHidden(bsz)
    enc_outs = Variable(torch.zeros(len_inp, bsz, encoder.nhid))
    enc_outs = enc_outs.cuda() if USE_CUDA else enc_outs
    for ei in range(len_inp):
        enc_outs[ei], enc_hid = encoder(var_inp[ei].unsqueeze(0), enc_hid)
    return enc_hid, enc_outs


def seq2seq_attn(var_inp, var_targ, encoder, decoder, teach=False):
    bsz = var_inp.size(1)
    len_targ = var_targ.size(0)
    outs = []
    attns = torch.zeros(bsz, len_targ, var_inp.size(0))

    enc_hid, enc_outs = encodeRNN(var_inp, encoder)
    dec_inp = var_inp[-1].unsqueeze(0)
    dec_hid = enc_hid
    for di in range(len_targ):
        dec_out, dec_hid, dec_attn = decoder(dec_inp, dec_hid, enc_outs)
        outs.append(dec_out)
        attns[:, di] = dec_attn.data
        if teach and random.random() < 0.5:
            dec_inp = var_targ[di].unsqueeze(0)
        else:
            dec_inp = dec_out
    return torch.cat(outs, 0), attns
