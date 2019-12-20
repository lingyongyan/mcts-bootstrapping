# coding=utf-8
""""
Program: ACER model
Description:
Author: Lingyong Yan
Date: 2018-10-11 16:01:57
Last modified: 2018-10-20 16:51:08
Python release: 3.6
Notes:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.attentions import GraphAttention, StateAttention


class ACER(Model):
    def __init__(self, args):
        super(ACER, self).__init__(args)
        self.hidden_dim = args.hidden_dim

        self.attn1 = GraphAttention(self.input_dim, 8, 8,
                                    n_head=self.hidden_dim // 2)
        self.attn2 = StateAttention(self.input_dim, self.hidden_dim, 8, 8,
                                    n_head=8)
        self.fc_a = nn.Linear(self.input_dim * self.hidden_dim,
                              self.output_dim)
        self.fc_c = nn.Linear(self.input_dim * self.hidden_dim,
                              self.output_dim)
        self._reset()

    def forward(self, inputs):
        pos_x = inputs[0]
        neg_x = inputs[1]
        pos_attn = self.attn1(pos_x, pos_x, pos_x)
        neg_attn = self.attn1(neg_x, neg_x, neg_x)
        x = torch.cat([pos_attn, neg_attn], dim=2)
        x_attn = self.attn2(x, x, x)
        x_fusion = x_attn.view(x_attn.size(0), -1)
        action_scores = self.fc_a(x_fusion)
        policy = F.softmax(action_scores, dim=-1).clamp(max=1 - 1e-5)
        Q = self.fc_c(x_fusion)
        value = (Q * policy).sum(-1, keepdim=True)
        return policy, Q, value
