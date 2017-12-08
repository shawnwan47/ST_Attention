import torch
import torch.nn as nn
from torch.autograd import Variable

import Layers

from Utils import aeq
from UtilClass import BottleLinear


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.attn = args.attn
        if self.attn:
            self.rnn = Layers.RNNAttn(
                args.rnn_type, args.input_size, args.hidden_size,
                args.num_layers, args.dropout, args.attn_type)
        else:
            self.rnn = Layers.RNNBase(
                args.rnn_type, args.input_size, args.hidden_size,
                args.num_layers, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.linear_out = nn.Linear(args.hidden_size, args.input_size)

    def forward(self, src, tgt, teach=True):
        srcL, bsz, ndim = src.size()
        tgtL, bsz_, ndim_ = tgt.size()
        aeq(bsz, bsz_)
        aeq(ndim, ndim_)
        hidden = self.rnn.initHidden(src)
        context, hidden = self.rnn.encode(src, hidden)
        if teach:
            if self.attn:
                output, hidden, context, attns = self.rnn(
                    tgt, hidden, context)
            else:
                output, hidden = self.rnn(tgt, hidden)
            output = tgt + self.linear_out(self.dropout(output))
        else:
            # inputfeeding prediction
            output = tgt.clone()
            attns = Variable(torch.zeros(tgtL, bsz, srcL + tgtL))
            input = tgt[0].unsqueeze(0)
            for i in range(tgtL):
                if self.attn:
                    res, hidden, context, attn = self.rnn(
                        input, hidden, context)
                    attns[i, :, :srcL + i] = attn[0, :, :-1]
                else:
                    res, hidden = self.rnn(input, hidden)
                output[i] = input + self.linear_out(self.dropout(res))
                input = output[i].clone().unsqueeze(0)
        if self.attn:
            return output, hidden, attns
        else:
            return output, hidden


class Transformer(nn.Module):
    """
    The Transformer decoder for Spatial-Temporal Attention Model
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Transformer, self).__init__()
        self.linear_in = BottleLinear(input_size, hidden_size)
        self.linear_out = BottleLinear(hidden_size, input_size)
        self.transformer_layers = nn.ModuleList(
            [Layers.TransformerLayer(hidden_size, dropout)
             for _ in range(num_layers)])

    def encode(self, input):
        output = self.linear_in(input)
        for layer in self.transformer_layers:
            output, attn = layer.encode(output)
        return output, attn

    def forward(self, input, context):
        # CHECKS
        aeq(input.dim(), 3)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END CHECKS
        output = self.linear_in(input)

        output = output.transpose(0, 1).contiguous()
        context = context.transpose(0, 1).contiguous()

        for layer in self.transformer_layers:
            output, attn = layer(output, context)

        # Process the result and update the attns.
        output = output.transpose(0, 1).contiguous()
        output = input + self.linear_out(output)
        return output, attn


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_layers = args.num_layers
        self.gc_in = Layers.GraphConvolution(args.past, args.hidden_size)
        self.gc_hid = Layers.GraphConvolution(args.hidden_size, args.hidden_size)
        self.gc_out = Layers.GraphConvolution(args.hidden_size, args.future)
        self.activation = nn.Sequential(nn.ReLU(), nn.Dropout(args.dropout))

    def forward(self, x, adj):
        x = self.activation(self.gc_in(x, adj))
        for l in range(self.num_layers - 2):
            x = self.activation(self.gc_hid(x, adj))
        return self.gc_out(x, adj)


class GAT(nn.Module):
    '''
    A simplified 2-layer GAT
    '''

    def __init__(self, input_size, hidden_size, head_count, output_size):
        super(GAT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.head_count = head_count

        self.gat1 = Layers.GraphAttention(input_size, hidden_size, head_count, True)
        self.gat2 = Layers.GraphAttention(self.gat1.output_size, output_size)

    def forward(self, input, adj):
        out, att1 = self.gat1(input, adj)
        out, att2 = self.gat2(out, adj)
        return out, att1, att2
