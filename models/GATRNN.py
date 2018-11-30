import torch
import torch.nn as nn

from models import GRNN
from models import GAT
from models.Framework import Seq2VecBase, Seq2SeqBase


class GARNN(DCRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.layers = nn.ModuleList([
            GARNNCell(rnn_type, size, gc_func, gc_kwargs)
            for i in range(num_layers)
        ])


class GARNNDecoder(GARNN):
    def __init__(self, rnn_type, num_nodes, size, out_size, num_layers, dropout,
                 gc_func, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         gc_func, gc_kwargs)
        self.gat = gc_func(input_size=size, output_size=1, **gc_kwargs)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        output, attn = self.gat(output)
        return output, hidden, attn
