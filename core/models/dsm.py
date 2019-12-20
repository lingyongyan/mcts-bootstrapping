# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-04 23:38:02
Last modified: 2019-02-22 17:31:44
Python release: 3.6
Notes:
"""


class DomainSemanticModel():
    def __init__(self, env, args):
        self.env = env
        self.logger = args.logger
        self.policy = args.policy

    def predict(self, state, actions):
        return self.policy.predict(self.env, state, actions)

    def reset(self):
        self.policy.reset()
