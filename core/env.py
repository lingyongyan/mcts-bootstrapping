# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-10-12 17:15:08
Last modified: 2019-09-01 21:42:48
Python release: 3.6
Notes:
"""
import math
import heapq
from core.components.state import State
from core.utils.torchutil import reward_weight


class BootEnv(object):
    def __init__(self, args, policy=None, env_id=0):
        self.logger = args.logger
        self.externals = args.externals
        self.storage = args.storage
        self.get_expanding_patterns = args.get_expanding_patterns
        self.sort_expanding_patterns = args.sort_expanding_patterns
        # self.exploit_sort = args.exploit_sort
        self.max_iters = args.max_iters
        self.max_entities = args.max_entities
        self.reward_model = args.reward_model
        self.policy = policy

    def step(self, state, actions, n_entities, adaption=1.0, exploit=False):
        for action in actions:
            if action < 0 or action >= len(state.actions):
                raise ValueError("action id should among valid action space")
        patterns = [state.actions[action] for action in actions]

        self.logger.info('use patterns: %s' % ','.join(patterns))
        new_entities, old_scores = self._expand_patterns(state, patterns,
                                                         exploit=exploit)
        entity_scores, pos = self._score_state(state, new_entities)
        self._expand_entities(state, new_entities, entity_scores, n_entities,
                              adaption=adaption, exploit=exploit)
        self._update_state(state)
        reward, stats = self._reward(entity_scores, old_scores)
        self.logger.info('\t[%d/%d] entities,reward:%.3f' %
                         (pos, len(entity_scores), reward))
        return state, stats

    def init_state(self, ps, ns, action_size, pos_threshold, only_top=False):
        state = State(ps, neg_seeds=ns, action_size=action_size,
                      pos_threshold=pos_threshold, logger=self.logger,
                      only_top=only_top)
        for seed in ps:
            state.expand_entity(seed, 1.0, self.storage, True, False)
        for seed in ns:
            state.expand_entity(seed, 0.0, self.storage, False, True)
        self._update_state(state)
        return state

    def is_ended(self, state):
        if len(state.pattern_history) >= self.max_iters or \
                len(state.top) >= self.max_entities:
            return 1
        return 0

    def _reward(self, entity_scores, old_scores):
        self.logger.debug('calc reward')
        new_sum = sum(entity_scores)
        old_sum = sum(old_scores)
        new_count = len(entity_scores)
        old_count = len(old_scores)
        '''
        new_count = len(entity_scores)
        total_score = new_score + sum(old_scores)
        all_count = new_count + len(old_scores)
        reward = new_score / new_count * reward_weight(new_count)
        estimate = total_score / all_count
        '''
        if new_count:
            reward = new_sum / new_count * reward_weight(new_count)
        else:
            reward = 0
        stats = (new_sum, new_count, old_sum, old_count)
        return reward, stats

    def _expand_patterns(self, state, patterns, exploit=False):
        state.use_patterns(patterns)
        new_entities = []
        old_scores = []
        for e in self.storage.get_entities_by_patterns(patterns).keys():
            if e in state.pos_seeds:
                old_scores.append(1.0)
            elif e in state.neg_seeds or e in state.bottom:
                old_scores.append(0.0)
            else:
                if e in state.top:
                    old_scores.append(state.extracted_entities[e])
                else:
                    new_entities.append(e)
        return new_entities, old_scores

    def _expand_entities(self, state, entities, entity_scores, n_entities,
                         adaption=1.0, exploit=False):
        self.logger.debug('expand entities')
        scored_entities = [(e, score)
                           for e, score in zip(entities, entity_scores)]
        neg_count = math.ceil(0.2 * len(entities))
        remain_count = max(len(entities) - n_entities, 0)
        neg_count = min(neg_count, len(state.pattern_history), remain_count)
        tops = heapq.nlargest(n_entities, scored_entities, key=lambda x: x[1])
        bottoms = heapq.nsmallest(neg_count, scored_entities,
                                  key=lambda x: x[1])
        tops = set([e for e, w in tops])
        bottoms = set([e for e, w in bottoms])

        for e, score in scored_entities:
            if exploit:
                score *= adaption
                state.expand_entity(e, score, self.storage, e in tops,
                                    e in bottoms)
            else:
                state.expand_entity(e, score, self.storage, e not in bottoms,
                                    e in bottoms)

    def _score_state(self, state, entities):
        self.logger.debug('score entities')
        scores = self.reward_model.score(self, state, entities)
        pos = 0
        for entity, score in zip(entities, scores):
            self.logger.debug('%s : %.3f' % (entity, score))
            if score > state.pos_threshold:
                pos += 1
        return scores, pos

    def _update_state(self, state):
        """

        :param state:

        """
        self.logger.debug('update state')
        self.policy.reset()
        '''
        actions = self.sort_expanding_patterns(
            self.policy, state, self.storage, num=state.action_size)
        if exploit:
            actions = self.exploit_sort(self.policy, state, self.storage,
                                        num=state.action_size)
            self.policy.reset()
        else:
        '''
        actions = self.sort_expanding_patterns(state, self.storage,
                                               num=state.action_size)
        state.actions = actions[0]
        state.priors = actions[1]
        state.update_counter()
        state.valid = True
