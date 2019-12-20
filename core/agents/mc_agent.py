# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-04 23:18:50
Last modified: 2019-08-25 15:40:07
Python release: 3.6
Notes:
"""
from core.agents.mcts import MCTS
from core.agents.rollout import Rollout
from core.agents.basic_agent import BasicAgent


adaptions = {0: 1.0,
             1: 1.0,
             2: 0.99,
             3: 0.95,
             4: 0.85}


class MCAgent(BasicAgent):
    def __init__(self, args, env_prototype, model_prototype,
                 memory_prototype=None):
        super(MCAgent, self).__init__(args, env_prototype,
                                      model_prototype, memory_prototype)
        self.rollout_args = args.rollout_args
        self.mcts_args = args.mcts_args
        rollout = Rollout(self.env, self.rollout_args)
        self.mcts = MCTS(self.env, self.model, rollout, self.mcts_args)
        self.env.policy = self.model

    def _one_step(self, state, actions=None):
        if not self.env.is_ended(state):
            valid_actions = state.actions
            depth = min(4, len(state.pattern_history))
            adaption = adaptions[depth]
            if actions is None:
                pi, feedbacks = self.mcts.get_action_values(state)
                actions = self.select_strategy(pi, self.n_patterns)
            else:
                feedbacks = [1.0] * len(valid_actions)
                pi = None
            self.logger.info('------Select pattern:%s------')
            for action in actions:
                self.logger.info('>>> %s' % valid_actions[action])

            self.weight_adaptive(state, feedbacks, valid_actions)
            state, _ = self.env.step(state, actions, self.n_entities,
                                     adaption=adaption, exploit=True)
            self.mcts.update_with_action(actions)
            state.update_gate()
            self.env.reward_model.reset()
        else:
            self.logger.info('env is ended')
        return state

    def weight_adaptive(self, state, feed_backs, actions):
        state.adjust_feedback(actions, feed_backs)
        self.env.reward_model.reset()
