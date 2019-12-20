# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-10-11 16:06:45
Last modified: 2018-10-25 21:37:00
Python release: 3.6
Notes:
"""

import torch
import torch.nn as nn

from core.attention import Attention
from core.utils import weight_init


class StateAttention(Attention):
    def __init__(self,
                 d_input,
                 d_key,
                 d_value,
                 n_head=10,
                 dropout=.1):
        super(StateAttention, self).__init__(d_input, d_key, d_value,
                                             n_head, dropout)
        self.v_layer = nn.Linear(d_input, d_value * self.n_head, bias=False)
        self.fc = nn.Linear(d_value * self.n_head, d_input)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.InstanceNorm1d(d_input)
        self.apply(weight_init)

    def preforward(self, query, key, value):
        Q = self.q_layer(query)
        K = self.k_layer(key)
        V = self.v_layer(value)
        Q = torch.cat(Q.split(split_size=self.d_key, dim=-1), dim=0)
        K = torch.cat(K.split(split_size=self.d_key, dim=-1), dim=0)
        V = torch.cat(V.split(split_size=self.d_value, dim=-1), dim=0)
        return Q, K, V

    def postforward(self, attn, query):
        restore_chunk_size = int(attn.size(0) / self.n_head)
        attn = torch.cat(
            attn.split(split_size=restore_chunk_size, dim=0), dim=-1)
        attn = self.dropout(self.fc(attn))
        output = self.norm((attn + query).transpose(-1, -2)).transpose(-1, -2)
        return output
