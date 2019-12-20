# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-15 14:34:44
Last modified: 2019-01-13 18:19:46
Python release: 3.6
Notes:
"""
import torch
import torch.nn as nn
from core.utils.torchutil import sequence_mask, mask_mean_weights


class CBOWEncoder(nn.Module):
    def __init__(self, vectors=None, vocab_size=None, emb_dim=None):
        super(CBOWEncoder, self).__init__()
        if vectors is not None:
            self.embed = nn.Embedding.from_pretrained(vectors)
        else:
            self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.requires_grad = False

    def forward(self, x, x_lens):
        embeddings = self.embed(x)
        masked_weights = mask_mean_weights(sequence_mask(x_lens))
        weighted_embedding = torch.bmm(masked_weights.unsqueeze(1), embeddings)
        return weighted_embedding.squeeze(1)
