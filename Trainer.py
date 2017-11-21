import random

import torch
from torch.autograd import Variable

from Constants import USE_CUDA


def train_seq2seq_attn(var_inp, var_targ, enc, dec, opt_enc, opt_dec, crit):
    bsz = var_inp.size(1)
    len_inp = var_inp.size(0)
    len_targ = var_targ.size(0)

    opt_enc.zero_grad()
    opt_dec.zero_grad()

    # encoding
    enc_hid = enc.initHidden(bsz)
    enc_outs = Variable(torch.zeros(len_inp, bsz, enc.nhid))
    enc_outs = enc_outs.cuda() if USE_CUDA else enc_outs
    for ei in range(len_inp):
        enc_outs[ei], enc_hid = enc(var_inp[ei].unsqueeze(0), enc_hid)

    # decoder
    dec_inp = var_inp[-1].unsqueeze(0)
    dec_hid = enc_hid

    loss = 0
    for di in range(len_targ):
        dec_out, dec_hid, dec_attn = dec(dec_inp, dec_hid, enc_outs)
        loss += crit(dec_out, var_targ[di].unsqueeze(0))

        use_teacher_forcing = True if random.random() < 0.5 else False
        if use_teacher_forcing:
            dec_inp = var_targ[di].unsqueeze(0)
        else:
            dec_inp = dec_out

    loss.backward()

    opt_enc.step()
    opt_dec.step()

    return loss.data[0] / len_targ


def seq2seqEncoder(var_inp, enc):
    bsz = var_inp.size(1)
    len_inp = var_inp.size(0)
    # encoding
    enc_hid = enc.initHidden(bsz)
    enc_outs = Variable(torch.zeros(len_inp, bsz, enc.nhid))
    enc_outs = enc_outs.cuda() if USE_CUDA else enc_outs
    for ei in range(len_inp):
        enc_outs[ei], enc_hid = enc(var_inp[ei].unsqueeze(0), enc_hid)
    return enc_hid, enc_outs


def seq2seq_attn(var_inp, enc, dec, len_targ=8):
    bsz = var_inp.size(1)
    dec_outs = torch.zeros(len_targ, bsz, dec.ndim)
    dec_attns = torch.zeros(len_targ, bsz, var_inp.size(0))
    enc_hid, enc_outs = seq2seqEncoder(var_inp, enc)
    dec_inp = var_inp[-1].unsqueeze(0)
    dec_hid = enc_hid
    for di in range(len_targ):
        dec_out, dec_hid, dec_attn = dec(dec_inp, dec_hid, enc_outs)
        print(dec_attn)
        dec_outs[di] = dec_out.data
        dec_attns[di] = dec_attn.transpose(0, 1).data
        dec_inp = dec_out
    return dec_outs, dec_attns
