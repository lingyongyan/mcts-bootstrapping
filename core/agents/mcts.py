# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-13 10:42:10
Last modified: 2019-08-26 10:06:48
Python release: 3.6
Notes:
"""
import copy
import math

import numpy as np

from core.utils.torchutil import weight_adaption, calc_reward


class TreeNode(object):
    def __init__(self, logger, parent, prior_p, cpuct, gamma):
        """
        """
        self.logger = logger
        self._parent = parent
        self._children = {}
        self._n_visit = 0
        self._Q = 0
        self.feedback = 0.
        self._max_Q = 0
        self.gamma = gamma
        self.cpuct = cpuct
        self._P = prior_p
        self.reward = None
        self.stats = None
        self.expanded = set()

    def expand(self, actions, action_priors, gamma):
        for action, prob in zip(actions, action_priors):
            if action not in self._children:
                self._children[action] = TreeNode(self.logger, self,
                                                  prob, self.cpuct, gamma)

    def select(self, is_exploration):
        if is_exploration:
            if self.is_root() and not self.is_all_expand():
                action = self.random_choose_new()
                assert action is not None
                self.expanded.add(action)
                return action, self._children[action]
            else:
                return self.best_child(is_exploration)
        else:
            return self.best_child(is_exploration)

    def backup(self, new_reward):
        self._n_visit += 1
        self._Q += (new_reward - self._Q) / self._n_visit
        if self._max_Q < new_reward:
            self._max_Q = new_reward

    def backup_recursive(self, following_stats):
        if self.is_root():
            self._n_visit += 1
            return
        new_stats = calc_reward(self.stats, self.gamma, following_stats)
        self.reward = new_stats[0]
        self.backup(self.reward)
        if self._parent:
            if self._parent.is_root():
                self.feedback += (self.reward - self.feedback)/self._n_visit
                self.logger.info('updated feedback is %.4f' % self._Q)
            self._parent.backup_recursive(new_stats)

    def best_child(self, is_exploration):
        if is_exploration:
            return max(self._children.items(), key=lambda x: x[1].get_value())
        else:
            return max(self._children.items(), key=lambda x: x[1]._Q)

    def get_value(self):
        if not self.is_root():
            _u = math.sqrt(self._parent._n_visit) / (self._n_visit + 1)
            _u = self.cpuct * self._P * _u
        else:
            _u = 0
        return self._Q + _u

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None

    def is_all_expand(self):
        if len(self.expanded) == len(self._children):
            return True
        return False

    def random_choose_new(self):
        nodes = set(self._children.keys()) - set(self.expanded)
        if len(nodes) > 0:
            return np.random.choice(list(nodes))
        else:
            return None


class MCTS(object):
    def __init__(self, env, model, rollout, args):
        self._root = TreeNode(args.logger, None, 1.0, args.cpuct, args.gamma)
        self.env = env
        self.model = model
        self.rollout = rollout
        self.args = args
        self.logger = args.logger
        self.cpuct = args.cpuct
        self.n_entities = args.n_entities

    def _playout(self, state, leaf_depth):
        node = self._root
        v = 0
        depth = 0
        root_flag = False
        while depth < leaf_depth:
            # Expand node if it has not already been done.
            if node.is_leaf():
                action_probs, v = self.model.predict(state, state.actions)
                node.expand(list(range(len(state.actions))),
                            action_probs, self.args.gamma)
                if node.is_root():
                    root_flag = True
                else:
                    break
            action, node = node.select(True)
            self.logger.info('--action "%s"; Prior:%.5f, N:%d, MQ:%.5f, '
                             'Q: %.4f, FB: %.4f'
                             % (state.actions[action], node._P, node._n_visit,
                                node._max_Q, node._Q, node.feedback))
            state, stats = self.env.step(state, [action], self.n_entities)
            node.stats = stats
            depth += 1
            if root_flag:
                break
        z = self.rollout.simulate(state, self.args.total_depth - depth)
        # leaf_value = (1 - self.args._lambda) * v + self.args._lambda * z
        # self.logger.info('leaf_reward is %f' % z[0])
        node.backup_recursive(z)

    def get_action_values(self, state):
        for n in range(self.args.sim_num):
            self.logger.info('Simulation for the %d time' % (n + 1))
            state_copy = copy.deepcopy(state)
            self._playout(state_copy, self.args.max_depth)
        weights = [(self._root._children[a]._n_visit,
                    self._root._children[a]._Q, a)
                   if a in self._root._children else 0
                   for a in range(len(state.actions))]
        feedbacks = [self._root._children[a].feedback
                     if a in self._root._children else 0
                     for a in range(len(state.actions))]
        return weight_adaption(weights), feedbacks

    def get_action(self, state):
        action_probs = self.get_action_prob(state)
        return max(action_probs, key=lambda x: x[1])[0]

    def update_with_action(self, last_actions):
        self._root = TreeNode(self.logger,
                              None, 1.0, self.args.cpuct, self.args.gamma)
