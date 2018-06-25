import torch
import torch.nn as nn


class Seq2VecBase(nn.Module):
    def __init__(self, embedding, encoder, decoder, history, horizon):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
        self.horizon = horizon


class Seq2VecRNN(Seq2VecBase):
    def forward(self, data, time, day):
        input = self.embedding(data, time, day)
        hidden, _ = self.encoder(input)
        output = self.decoder(hidden[:, -1])
        return output

class Seq2VecRNNAttn(Seq2VecBase):
    def forward(self, data, time, day):
        input = self.embedding(data, time, day)
        hidden, _ = self.encoder(input)
        output, attention = self.decoder(hidden[:, [-1]], hidden)
        return output, attention


class Seq2VecDCRNN(Seq2VecBase):
    def forward(self, data, time, day):
        input = self.embedding(data.unsqueeze(-1), time, day)
        hidden, _ = self.encoder(input)
        output = self.decoder(hidden[:, -1])
        return output


class Seq2VecGARNN(Seq2VecBase):
    def forward(self, data, time, day):
        input = self.embedding(data.unsqueeze(-1), time, day)
        hidden, _, attn_i, attn_h = self.encoder(input)
        output = self.decoder(hidden[:, -1])
        return output, attn_i[:, -1], attn_h[:, -1]


class Seq2VecTransformer(Seq2VecBase):
    def forward(self, data, time, day):
        input = self.embedding(data.unsqueeze(-1), time, day)
        hidden, attention = self.encoder(input)
        output = self.decoder(hidden[:, -1])
        return output, attention[:, -1]


def Seq2VecSTTransformer(Seq2VecBase):
    def forward(self, data, time, day):
        input = self.embedding(data.unsqueeze(-1), time, day)
        hidde, attn_s, attn_t = self.encoder(input)
        output = self.decoder(hidden[:, -1])
        return output, attn_s[:, -1], attn_t[:, -1]


def build_Seq2Vec(args):
    if args.model == 'RNN':
        embedding = Embedding.build_temp_embedding(args)
        encoder = RNN.build_RNN(args)
        decoder = Decoder.build_temp_vec(args)
        seq2vec = Seq2VecRNN
    elif args.model == 'DCRNN':
