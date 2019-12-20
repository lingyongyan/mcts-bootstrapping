# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2019-01-10 13:30:43
Last modified: 2019-01-10 15:22:37
Python release: 3.6
Notes:
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def rnn_forwarder(rnn, inputs, seq_lengths, hidden):
    batch_first = rnn.batch_first
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    _, desorted_indices = torch.sort(indices, descending=False)

    if batch_first:
        inputs = inputs[indices]
    else:
        inputs = inputs[:, indices]
    packed_inputs = pack_padded_sequence(inputs, sorted_seq_lengths,
                                         batch_first=batch_first)
    outputs, hidden_ = rnn(packed_inputs, hidden)
    padded_res, _ = pad_packed_sequence(outputs, batch_first=batch_first)
    if batch_first:
        desorted_outputs = padded_res[desorted_indices]
        hidden_ = hidden_[:, desorted_indices]
    else:
        desorted_outputs = padded_res[:, desorted_indices]
        hidden_ = hidden_[:, desorted_indices]
    return desorted_outputs, hidden_


class RNNEncoder(nn.Module):
    def __init__(self, num_layers, dropout=0.5, vectors=None,
                 emb_dim=None, vocab_size=None):
        super(RNNEncoder, self).__init__()

        self.rnn = nn.GRU(input_size=300, hidden_size=300, num_layers=2,
                          dropout=dropout, batch_first=True)
        assert vectors is not None or \
            emb_dim is not None and vocab_size is not None

        if vectors is not None:
            self.embed = nn.Embedding.from_pretrained(vectors)
        else:
            self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.requires_grad = False

    def forward(self, x, x_len):
        x_emb = self.embed(x)
        output, _ = rnn_forwarder(self.rnn, x_emb, x_len, None)
        ind = (x_len - 1).view(-1, 1).expand(x_emb.size(0), x_emb.size(2))
        ind = ind.unsqueeze(1)
        output = output.gather(1, ind).squeeze(1)
        return output
