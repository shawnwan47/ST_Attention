"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from Utils import aeq
from Layers import PositionwiseFeedForward
import MultiHeadedAttention


MAX_SIZE = 240


class TransformerEncoderLayer(nn.Module):
    def __init__(self, size, dropout, head_count=8, hidden_size=None):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerEncoderLayer, self).__init__()
        hidden_size = size if hidden_size is None else hidden_size

        self.self_attn = MultiHeadedAttention(head_count, size, p=dropout)
        self.feed_forward = PositionwiseFeedForward(size, hidden_size, dropout)

    def forward(self, input, mask=None):
        mid, _ = self.self_attn(input, input, input, mask=mask)
        out = self.feed_forward(mid)
        return out


class TransformerEncoder(nn.Module):
    """
    The Transformer encoder from "Attention is All You Need".
    """

    def __init__(self, num_layers, hidden_size, dropout):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, dropout)
             for i in range(num_layers)])

    def forward(self, input):
        out = input.transpose(0, 1).contiguous()
        for i in range(self.num_layers):
            out = self.transformer[i](out)
        return Variable(input.data), out.transpose(0, 1).contiguous()


class TransformerDecoderLayer(nn.Module):
    def __init__(self, size, dropout, head_count=8, hidden_size=None):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerDecoderLayer, self).__init__()
        hidden_size = size if hidden_size is None else hidden_size
        self.self_attn = MultiHeadedAttention(head_count, size, p=dropout)
        self.context_attn = MultiHeadedAttention(head_count, size, p=dropout)
        self.feed_forward = PositionwiseFeedForward(size, hidden_size, dropout)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        self.register_buffer('mask', mask)

    def forward(self, input, context, src_mask, tgt_mask):
        # Args Checks
        input_batch, input_len, _ = input.size()
        contxt_batch, contxt_len, _ = context.size()
        aeq(input_batch, contxt_batch)

        src_batch, t_len, s_len = src_mask.size()
        tgt_batch, t_len_, t_len__ = tgt_mask.size()
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        aeq(t_len, t_len_, t_len__, input_len)
        aeq(s_len, contxt_len)
        # END Args Checks

        dec_mask = torch.gt(tgt_mask + self.mask[:, :t_len_, :t_len_], 0)
        query, attn = self.self_attn(input, input, input, mask=dec_mask)
        mid, attn = self.context_attn(context, context, query, mask=src_mask)
        output = self.feed_forward(mid)

        # CHECKS
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(contxt_batch, output_batch)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(contxt_len, s_len_)
        aeq(input_len, t_len_)
        # END CHECKS

        return output, attn

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder for Spatial-Temporal Attention Model
    """

    def __init__(self, num_layers, hidden_size, dropout):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, dropout)
             for _ in range(num_layers)])

    def forward(self, input, context):
        """
        Forward through the TransformerDecoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                                of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # CHECKS
        aeq(input.dim(), 3)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END CHECKS

        # Run the forward pass of the TransformerDecoder.

        output = input.transpose(0, 1).contiguous()
        src_context = context.transpose(0, 1).contiguous()

        for i in range(self.num_layers):
            output, attn = self.transformer_layers[i](output, src_context)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        return outputs, attn
