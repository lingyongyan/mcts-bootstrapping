# coding=utf-8
""""
Program: This is the base model of Agent to do basic settings and declairing.
Description:
Author: Lingyong Yan
Date: 2018-10-22 13:45:39
Last modified: 2019-08-24 23:13:59
Python release: 3.6
Notes:
"""
import os
from core.utils.utils import Transition


class Brain(object):
    def __init__(self, args, env_prototype, model_prototype):
        self.logger = args.logger
        self.env_prototype = env_prototype
        self.env_params = args.env_params
        self.model_prototype = model_prototype
        self.model_params = args.model_params
        self.cache_path = args.cache_path

    def _reset_transition(self):
        self.transition = Transition(state=None,
                                     action=None,
                                     reward=None,
                                     next_state=None,
                                     done=False)

    def _forward(self, state):
        raise NotImplementedError("not implemented in base calss")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base calss")
