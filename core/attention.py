# coding=utf-8
""""
Program: :Attention Module
Description:
Author: Lingyong Yan
Date: 2018-09-07 10:04:51
Last modified: 2018-10-11 16:33:43
Python release: 3.6
Notes:
"""

import torch
import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    def __init__(self,
                 d_input,
                 d_key,
                 d_value,
                 n_head=10,
                 dropout=.1):
        super(Attention, self).__init__()
        self.d_input = d_input
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.dropout = dropout
        self.attention = ScaledDotProduct(temperature=np.power(d_key, 0.5),
                                          dropout=dropout)
        self.q_layer = nn.Linear(d_input, d_key * self.n_head, bias=False)
        self.k_layer = nn.Linear(d_input, d_key * self.n_head, bias=False)

    def forward(self, query, key, value):
        Q, K, V = self.preforward(query, key, value)
        attn = self.attention(Q, K, V)
        attn = self.postforward(attn, query)
        return attn

    def preforward(self, query, key, value):
        return query, key, value

    def postforward(self, attn, query):
        return attn


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature, dropout=0.0):
        super(ScaledDotProduct, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        attn = torch.matmul(Q, K.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output
