import torch.nn as nn
from modules import GraphGRUModel


class IsoRNN(GraphGRUModel):
    def __init__(self, embedding, framework, rnn_attn, horizon,
                 model_dim, num_layers, dropout):
        super().__init__(
            embedding=embedding,
            horizon=horizon,
            framework=framework,
            rnn_attn=rnn_attn,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
            func=nn.Linear,
            func_kwargs={}
        )
