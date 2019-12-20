# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-18 17:21:18
Last modified: 2019-09-01 00:47:00
Python release: 3.6
Notes:
"""
import os
from core.brain import Brain


class BasicAgent(Brain):
    def __init__(self, args, env_prototype, model_prototype,
                 memory_prototype=None):
        super(BasicAgent, self).__init__(args, env_prototype, model_prototype)
        self.ps = args.ps
        self.ns = args.ns
        self.pos_threshold = args.pos_threshold
        self.action_size = args.action_size
        self.n_patterns = args.n_patterns
        self.n_entities = args.n_entities
        self.only_top = args.only_top
        self.select_strategy = args.select_strategy
        self.env = self.env_prototype(self.env_params)
        self.model = self.model_prototype(self.env, self.model_params)

    def _one_step(self, state, action=None):
        raise NotImplementedError("Not implemented in base calss")

    def fit_model(self):
        for i in range(1, self.args.num_iter + 1):
            self.logger.info('-------Iteration ' + str(i) + '------')

    def test_model(self):
        state = self.init_state()
        step = 1
        self.logger.info('Seed state:')
        self.logger.info(state.pos_seeds)
        state.to_json_file(os.path.join(self.cache_path, 'step_0.json'))
        while not self.env.is_ended(state):
            self.logger.info('-------Step ' + str(step) + '------')
            state = self._one_step(state)
            state.to_json_file(os.path.join(self.cache_path,
                                            'step_%d.json' % step))
            step += 1
        # self.env.extractor.miner.close()
        return state

    def init_state(self, ps=None, ns=None, action_size=None,
                   pos_threshold=None, only_top=None):
        ps = ps if ps else self.ps
        ns = ns if ns else self.ns
        action_size = action_size if action_size else self.action_size
        pos_threshold = pos_threshold if pos_threshold else self.pos_threshold
        only_top = only_top if only_top is not None else self.only_top

        state = self.env.init_state(ps=ps, ns=ns, action_size=action_size,
                                    pos_threshold=pos_threshold,
                                    only_top=only_top)
        return state
