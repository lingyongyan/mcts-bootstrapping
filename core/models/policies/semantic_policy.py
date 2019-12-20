# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-18 13:28:30
Last modified: 2019-08-26 09:31:53
Python release: 3.6
Notes:
"""
from functools import partial

import numpy as np

from core.models.policy import Policy


def get_pattern_embs(encoder, env, state, patterns):
    pattern_embs = encoder.encode(
        patterns, env.storage, state, type_='pattern')
    return pattern_embs


class SemanticPolicy(Policy):
    def __init__(self, encoder, sim, logger):
        super(SemanticPolicy, self).__init__(logger)
        self.encoder = encoder
        self.sim = sim
        self.core_emb = None

    def predict(self, env, state, actions):
        if not actions:
            return [], 0
        self.logger.debug('calculate pattern probs')
        if self.core_emb is None:
            self.logger.debug('calculate pattern core embs')
            core_entities = state.get_core_entities()
            self.core_emb = self.encoder.encode(
                core_entities, env.storage, state, type_='entity_set')
        self.logger.debug('calculate pattern sim')
        embeder = partial(get_pattern_embs, self.encoder, env, state)
        scores = self.sim.get_sim(embeder, self.core_emb, actions)
        if len(scores) > 0:
            # Mapping to [0,1] and then normalizing
            scores = 0.5 + 0.5 * scores
            scores = (scores / scores.sum()).tolist()
        else:
            scores = [1. / state.action_size] * state.action_size
        self.logger.debug('end')
        return scores, 0

    def reset(self):
        self.core_emb = None
