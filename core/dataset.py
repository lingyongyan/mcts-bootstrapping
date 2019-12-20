# coding=utf-8
""""
Program: Dataset
Description:
Author: Lingyong Yan
Date: 2018-07-23 23:09:06
Last modified: 2018-10-09 10:36:45
Python release: 3.6
"""
import torch

from .settings import device


class InstanceData(object):
    def __init__(self, func):
        self.func = func
        self.dataset = []
        self.labels = []

    def clear(self):
        self.dataset.clear()
        self.labels.clear()

    def add_instance(self, externals, pse, nse, pattern_pool, instance):
        features = self.func(externals,
                             pse,
                             nse,
                             pattern_pool,
                             instance)
        vec = torch.Tensor(features).to(device)
        label = torch.Tensor(externals.label(instance)).to(device)
        self.dataset.append(vec)
        self.labels.append(label)

    def add_instances(self, externals, pse, nse, pattern_pool, instances):
        for instance in instances:
            self.add_instance(externals,
                              pse,
                              nse,
                              pattern_pool,
                              instance)

    def get_tensor(self):
        x_tensor = torch.cat(self.dataset, dim=0).view(-1, 4)
        y_tensor = torch.cat(self.labels, dim=0).view(-1, 1)
        return x_tensor, y_tensor
