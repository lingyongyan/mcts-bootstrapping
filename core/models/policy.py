# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-18 13:20:48
Last modified: 2019-02-22 17:25:03
Python release: 3.6
Notes:
"""


class Policy(object):
    def __init__(self, logger):
        self.logger = logger

    def predict(self, env, state, actions):
        raise NotImplementedError("Not implemented in base class")

    def reset(self):
        raise NotImplementedError("Not implemented in base class")
