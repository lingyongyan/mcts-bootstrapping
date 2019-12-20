# coding=utf-8
""""
Program: Attention for long graph
Description:
Author: Lingyong Yan
Date: 2018-10-11 16:05:41
Last modified: 2018-10-17 17:20:02
Python release: 3.6
Notes:
"""

import torch
import torch.nn as nn

from core.attention import Attention
from core.utils import weight_init


class GraphAttention(Attention):
    def __init__(self,
                 d_input,
                 d_key,
                 d_value,
                 n_head=10,
                 dropout=.1):
        super(GraphAttention, self).__init__(d_input, d_key, d_value,
                                             n_head, dropout)
        self.layer_norm = nn.LayerNorm((self.n_head, self.d_input))
        self.apply(weight_init)

    def preforward(self, query, key, value):
        query = query.transpose(-1, -2).mean(-2, keepdim=True)
        key = key.transpose(-1, -2)
        value = value.transpose(-1, -2)
        Q = self.q_layer(query)
        K = self.k_layer(key)
        V = value
        Q = torch.cat(Q.split(split_size=self.d_key, dim=-1), dim=0)
        K = torch.cat(K.split(split_size=self.d_key, dim=-1), dim=0)
        return Q, K, V

    def postforward(self, attn, query):
        restore_chunk_size = int(attn.size(0) / self.n_head)
        attn = torch.cat(
            attn.split(split_size=restore_chunk_size, dim=0), dim=1)
        attn = self.layer_norm(attn).transpose(-1, -2)
        return attn
