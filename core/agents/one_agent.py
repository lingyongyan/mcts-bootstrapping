# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-04 23:18:50
Last modified: 2019-08-25 15:40:45
Python release: 3.6
Notes:
"""
from core.agents.basic_agent import BasicAgent


adaptions = {0: 1.0,
             1: 1.0,
             2: 0.99,
             3: 0.95,
             4: 0.85}


class OneAgent(BasicAgent):
    def __init__(self, args, env_prototype, model_prototype,
                 memory_prototype=None):
        super(OneAgent, self).__init__(args, env_prototype,
                                       model_prototype, memory_prototype)
        self.env.policy = self.model

    def _one_step(self, state, actions=None):
        if not self.env.is_ended(state):
            valid_actions = state.actions
            depth = min(4, len(state.pattern_history))
            adaption = adaptions[depth]
            if actions is None:
                pi, _ = self.model.predict(state, state.actions)
                actions = self.select_strategy(pi, self.n_patterns)

            self.logger.info('------Select pattern:%s------')
            for action in actions:
                self.logger.info('>>> %s' % valid_actions[action])
            state, _ = self.env.step(state, actions, self.n_entities,
                                     adaption=adaption, exploit=True)
            self.logger.info('New state:')
            self.logger.info(state.get_string_representation())
        else:
            self.logger.info('env is ended')
        return state
