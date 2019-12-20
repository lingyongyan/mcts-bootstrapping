# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-19 20:40:44
Last modified: 2019-05-20 09:23:26
Python release: 3.6
Notes:
"""
from core.models.encoder import Encoder


class BasicEncoder(Encoder):

    def encode(self, patterns, storage=None,
               state=None, type_='pattern'):
        embeddings = self._embedding(patterns)
        return embeddings
