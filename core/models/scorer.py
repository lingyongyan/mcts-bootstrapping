# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-18 13:18:58
Last modified: 2019-02-22 17:22:17
Python release: 3.6
Notes:
"""


class Scorer(object):
    def __init__(self, logger):
        self.logger = logger

    def score(env, state, entities):
        raise NotImplementedError("Not implemented in base class")

    def reset(self):
        raise NotImplementedError("Not implemented in base class")
