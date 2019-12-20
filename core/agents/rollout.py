# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-14 13:00:27
Last modified: 2019-08-25 16:12:39
Python release: 3.6
Notes:
"""
import numpy as np
from core.utils.torchutil import discounted_reward


class Rollout(object):
    def __init__(self, env, args):
        self.env = env
        self.policy = args.policy
        self.gamma = args.gamma
        self.logger = args.logger
        self.n_entities = args.n_entities

    def simulate(self, state, depth):
        stats_list = []
        for i in range(depth):
            if self.env.is_ended(state):
                break
            self.logger.info('--quick rollout at %d depth' % (i + 1))
            action_probs, _ = self.policy.predict(
                self.env, state, state.actions)
            '''
            action_probs = state.priors
            '''
            action = np.argmax(action_probs)
            new_state, stats = self.env.step(state, [action], self.n_entities)
            state = new_state
            stats_list.append(stats)
        if len(stats_list) == 0:
            return (0, 0, 0)
        rewards = discounted_reward(stats_list, self.gamma)
        return rewards[0]
