import torch
import torch.nn as nn

from models import GRNNBase
from models.Framework import Seq2VecBase, Seq2SeqBase


class DiffusionConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj, hops):
        super().__init__()
        # calculate adjs
        self.linear = nn.Linear(input_size, output_size)
        self.filters = self._gen_adj_hops(adj, hops)
        self.filters += self._gen_adj_hops(adj.t(), hops)
        self.linears = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=False)
            for _ in self.filters
        ])

    @staticmethod
    def _gen_adj_hops(adj, hops):
        adj_norm = adj.div(adj.sum(1).unsqueeze(-1))
        adjs = [adj_norm]
        for _ in range(hops - 1):
            adjs.append(adjs[-1].matmul(adj_norm))
        return adjs

    def forward(self, input):
        output = self.linear(input)
        for linear, filter in zip(self.linears, self.filters):
            output += filter.matmul(linear(input))
        return output


class DCRNN(GRNNBase.GRNN):
    def __init__(self, rnn_type, size, num_layers, num_nodes, adj, hops):
        super().__init__(rnn_type, size, num_layers, num_nodes,
                         func=DiffusionConvolution,
                         adj=adj,
                         hops=hops)


class DCRNNDecoder(DCRNN):
    def __init__(self, rnn_type, size, num_layers, num_nodes, adj, hops):
        super().__init__(rnn_type, size, num_layers, num_nodes, adj, hops)
        self.linear_out = nn.Linear(size, 1)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        output = self.linear_out(output)
        return output, hidden


class DCRNNSeq2Seq(Seq2SeqBase):
    def forward(self, data, time, day, teach=0):
        data = data.unsqueeze(-1)
        self._check_args(data, time, day)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        input = self._expand_input0(input)
        output_i, hidden = self.decoder(input, hidden)
        output = [output_i]
        for idx in range(his, his + self.horizon - 1):
            # data_i = data[:, [idx]] if random() < teach else output_i.detach()
            data_i = output_i.detach()
            input = self.embedding(data_i, time[:, [idx]], day[:, [idx]])
            output_i, hidden = self.decoder(input, hidden)
            output.append(output_i)
        output = torch.cat(output, 1).squeeze(-1)
        return output
