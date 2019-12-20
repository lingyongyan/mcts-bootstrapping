# coding=utf-8
""""
Program: basic model for DRL
Description:
Author: Lingyong Yan
Date: 2018-10-11 15:50:03
Last modified: 2018-10-25 13:42:31
Python release: 3.6
Notes:
"""
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.logger = args.logger

        self.use_cuda = args.use_cuda
        if hasattr(args, 'enable_lstm'):
            self.enable_lstm = args.enable_lstm

        self.device = args.device
        self.dtype = args.dtype

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim + 1

    def _init_weights(self):
        raise NotImplementedError('Not implemented in base class')

    def print_model(self):
        self.logger.debug('<------------------------> Model')
        self.logger.debug(self)

    def _reset(self):
        self._init_weights()
        # self.type(self.dtype)
        self.print_model()

    def forward(self, input):
        raise NotImplementedError('Not implemented in base class')
