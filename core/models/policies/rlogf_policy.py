# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-18 13:33:11
Last modified: 2019-02-24 00:44:42
Python release: 3.6
Notes:
"""
from core.models.policy import Policy
from core.utils.torchutil import weight_normalize


class RlogfPolicy(Policy):
    def __init__(self, logger):
        super(RlogfPolicy, self).__init__(logger)

    def predict(self, env, state, actions):
        weights = weight_normalize(state.priors)
        return weights.tolist(), 0

    def reset(self):
        pass
