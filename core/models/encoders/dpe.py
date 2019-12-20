# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2019-02-16 21:14:07
Last modified: 2019-08-25 04:49:38
Python release: 3.6
Notes:
"""
from core.utils.torchutil import weight_normalize
from core.components.filters import get_adpe


class DPE(object):
    def __init__(self,):
        self.cache = {}

    def dsm_init_weight(self, storage, state, entities, context_count,
                        type_='entity'):
        if type_ == 'entity_set':
            return self.get_adpe(storage, state, entities, context_count)
        else:
            return self.get_dpe(storage, state, entities, context_count)

    def get_adpe(self, storage, state, core_entities, context_count):
        adaptions = state.adaptions if state else None
        patterns, weights = get_adpe(storage, core_entities,
                                     context_count=context_count,
                                     adaptions=adaptions)
        weights = weight_normalize(weights)
        result = (patterns, weights)
        return [result]

    def get_dpe(self, storage, state, entities, context_count):
        pattern_weights = []
        for entity in entities:
            patterns, weights = get_adpe(storage, [entity],
                                         context_count=context_count,
                                         adaptions=state.adaptions)
            weights = weight_normalize(weights)
            result = (patterns, weights)
            pattern_weights.append(result)
        return pattern_weights
