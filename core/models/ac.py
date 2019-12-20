# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-10-20 16:51:22
Last modified: 2018-10-28 21:54:58
Python release: 3.6
Notes:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.utils import weight_init
from core.attentions.state_attention import StateAttention


class ActorCritic(Model):
    def __init__(self, args):
        super(ActorCritic, self).__init__(args)
        self.hidden_dim = args.hidden_dim

        self.ln = nn.InstanceNorm1d(self.input_dim[1], affine=True)

        self.fc1 = nn.Linear(self.input_dim[1], 128)
        self.fc2 = nn.Linear(128, 64)

        self.attn = StateAttention(64, 8, 8)

        self.fc3 = nn.Linear(64, 1)

        self.value_layer = nn.Linear(self.output_dim, 1)

        if self.enable_lstm:
            self.lstm = nn.LSTMCell(self.input_dim[0], 1)

        self._reset()

    def _init_weights(self):
        self.apply(weight_init)

        if self.enable_lstm:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    def forward(self, inputs, lstm_hidden=None):
        batch_size = inputs.size(0)
        if lstm_hidden:
            assert batch_size == lstm_hidden[0].size(0)
        x = self.ln(inputs.transpose(-1, -2)).transpose(-1, -2).contiguous()
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.attn(x, x, x).contiguous()
        x = F.tanh(self.fc3(x))
        x = x.view(batch_size, -1)
        if self.enable_lstm:
            zero, c = self.lstm(x, lstm_hidden)
        else:
            zero = torch.zeros(batch_size, 1)
        x_all = torch.cat([zero, x], dim=-1)
        policy = F.softmax(x_all, dim=-1).clamp(max=1 - 1e-4)
        value = self.value_layer(x_all)
        if self.enable_lstm:
            return policy, value, (zero, c)
        else:
            return policy, value
