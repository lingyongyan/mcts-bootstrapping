# coding=utf-8
""""
Program: bootstrapping environment
Description:
Author: Lingyong Yan
Date: 2018-12-03 23:36:36
Last modified: 2019-05-20 08:27:40
Python release: 3.6
Notes:
"""
import heapq
from core.env import Env
from core.components.state import State
from core.utils.torchutil import reward_weight


class BootEnv(Env):
    def __init__(self, args, policy=None, env_id=0):
        super(BootEnv, self).__init__(args, env_id)
        self.externals = args.externals
        # self.extractor = args.extractor
        self.storage = args.storage
        self.sort_expanding_patterns = args.sort_expanding_patterns
        self.get_expanding_patterns = args.get_expanding_patterns
        self.exploit_sort = args.exploit_sort
        self.max_iters = args.max_iters
        self.max_entities = args.max_entities
        self.top_entities = args.top_entities
        self.reward_model = args.reward_model
        self.policy = policy

    def step(self, state, actions, exploit=False, adaption=1.0):
        for action in actions:
            if action < 0 or action >= len(state.actions):
                raise ValueError("action id should among valid action space")
        patterns = [state.actions[action] for action in actions]

        self.logger.info('use patterns: %s' % ','.join(patterns))
        new_entities, old_scores = self._expand_patterns(state, patterns,
                                                         exploit=exploit)
        entity_scores, pos = self._score_state(state, new_entities)
        self._expand_entities(state, new_entities, entity_scores,
                              exploit=exploit, adaption=adaption)
        self._update_state(state, exploit=exploit)
        reward, estimate, new_count, all_count = self._reward(entity_scores,
                                                              old_scores)
        self.logger.info('\t[%d/%d] entities,reward:%.3f, score:%.3f' %
                         (pos, new_count, reward, estimate))
        return state, reward, estimate

    def init_state(self, ps, ns, action_size, pos_threshold):
        state = State(ps, neg_seeds=ns, action_size=action_size,
                      pos_threshold=pos_threshold)
        ps_scores = [1.0] * len(ps)
        self._expand_entities(state, ps, ps_scores)
        self._update_state(state, exploit=True)
        return state

    def is_ended(self, state):
        core_entities = state.get_core_entities()
        if len(state.pattern_history) >= self.max_iters or \
                len(core_entities) >= self.max_entities:
            return 1
        return 0

    def _reward(self, entity_scores, old_scores):
        self.logger.debug('calc reward')
        if len(entity_scores) == 0:
            return 0, 0, 0, len(old_scores)
        new_score = sum(entity_scores)
        total_score = new_score + sum(old_scores)
        new_count = len(entity_scores)
        all_count = new_count + len(old_scores)
        reward = new_score / new_count * reward_weight(new_count)
        estimate = total_score / all_count
        return reward, estimate, new_count, all_count

    def _expand_patterns(self, state, patterns, exploit=False):
        for p in patterns:
            state.use_pattern(p)
        new_entities = []
        old_scores = []
        for e in self.storage.get_entities_by_patterns(patterns).keys():
            if e in state.pos_seeds or e in state.pos_extracted:
                old_scores.append(1.0)
            elif e in state.neg_seeds:
                old_scores.append(0.0)
            else:
                if exploit:
                    if e in state.top:
                        old_scores.append(state.extracted_entities[e])
                    else:
                        new_entities.append(e)
                else:
                    if e in state.extracted_entities:
                        old_scores.append(state.extracted_entities[e])
                    else:
                        new_entities.append(e)
        return new_entities, old_scores

    def _expand_entities(self, state, entities, entity_scores,
                         exploit=False, adaption=1.0):
        self.logger.debug('expand entities')
        scored_entities = [(e, score)
                           for e, score in zip(entities, entity_scores)]
        tops = heapq.nlargest(self.top_entities, scored_entities,
                              key=lambda x: x[1])
        top_entities = set([e for e, w in tops])

        for e, score in scored_entities:
            if exploit:
                score *= adaption
            state.expand_entity(e, score, self.storage, e in top_entities)

    def _score_state(self, state, entities):
        self.logger.debug('score entities')
        scores = self.reward_model.score(self, state, entities)
        pos = 0
        for entity, score in zip(entities, scores):
            self.logger.debug('%s : %.3f' % (entity, score))
            if score > state.pos_threshold:
                pos += 1
        return scores, pos

    def _update_state(self, state, exploit=False):
        """

        :param state:

        """
        self.logger.debug('update state')
        self.policy.reset()
        '''
        actions = self.sort_expanding_patterns(
            self.policy, state, self.storage, num=state.action_size)
        '''
        if exploit:
            '''
            actions = self.exploit_sort(state, self.storage, num=state.action_size)
            '''
            actions = self.exploit_sort(self.policy, state, self.storage,
                                        num=state.action_size)
            self.policy.reset()
        else:
            actions = self.sort_expanding_patterns(
                state, self.storage, num=state.action_size)
        state.actions = actions[0]
        state.priors = actions[1]
        state.update_counter()
        state.valid = True
