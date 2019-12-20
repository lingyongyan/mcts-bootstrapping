# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-18 13:23:07
Last modified: 2019-08-26 09:29:22
Python release: 3.6
Notes:
"""
from functools import partial
from core.models.scorer import Scorer


'''
def _inner_new(target_emb, emb):
    sim_matrix = cosine_similarity(target_emb, emb)
    sim_mean = sim_matrix.mean(dim=0)
    indices = sim_mean.topk(100)[1]
    return indices


def re_weights(target_lembs, target_rembs,
               l_embs, l_weights, r_embs, r_weights):
    target_lemb = target_lembs.squeeze(0)
    target_remb = target_rembs.squeeze(0)
    new_lembs = []
    new_lweights = []
    new_rembs = []
    new_rweights = []
    for le, lw, re, rw in zip(l_embs, l_weights, r_embs, r_weights):
        l_indices = _inner_new(target_lemb, le)
        new_lembs.append(le[l_indices, :])
        new_lweight = lw[l_indices]
        new_lweight /= new_lweight.sum()
        new_lweights.append(new_lweight)
        r_indices = _inner_new(target_remb, re)
        new_rembs.append(re[r_indices, :])
        new_rweight = rw[r_indices]
        new_rweight /= new_rweight.sum()
        new_rweights.append(new_rweight)
    return torch.stack(new_lembs, dim=0), torch.stack(new_lweights, dim=0),\
        torch.stack(new_rembs, dim=0), torch.stack(new_rweights, dim=0)
'''


def get_entity_embs(encoder, env, state, entities):
    entity_embs = encoder.encode(
        entities, env.storage, state, type_='entity')
    return entity_embs


class EmbScorer(Scorer):
    def __init__(self, encoder, sim, logger, type_):
        super(EmbScorer, self).__init__(logger)
        self.encoder = encoder
        self.sim = sim
        self.type_ = type_
        self.core_emb = None

    def score(self, env, state, entities):
        if not entities:
            return []
        self.logger.debug('calculate entity values')
        if self.core_emb is None:
            self.logger.info('calculate entity core embs')
            core_entities = state.get_core_entities()
            self.core_emb = self.encoder.encode(
                core_entities, env.storage, state, type_='entity_set')

        self.logger.debug('calculate entity sim')
        embeder = partial(get_entity_embs, self.encoder, env, state)
        scores = self.sim.get_sim(embeder, self.core_emb, entities)
        scores = scores.tolist()
        self.logger.debug('end')
        return scores

    def reset(self):
        self.core_emb = None
        self.sim.cache = {}
