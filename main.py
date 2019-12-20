# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-10-17 16:51:54
Last modified: 2019-12-20 09:11:01
Python release: 3.6
Notes:
"""
import torch
import numpy as np
import argparse

from core.params import AgentParams
from core.env import BootEnv
from core.agents.mc_agent import MCAgent
from core.agents.one_agent import OneAgent
from core.models.dsm import DomainSemanticModel


def regist_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./caches/corpus_storage.pkl',
                        help='preprocessed dataset')
    parser.add_argument('--entity_type', default='capital',
                        help='entity_type to be extracted')
    parser.add_argument('--n_patterns', type=int, default=5,
                        help='number of pattern to be selected')
    parser.add_argument('--n_entities', type=int, default=10,
                        help='number of entities to be expanded')
    parser.add_argument('--n_simulations', type=int, default=500,
                        help='MCTS algorithm simulation numbers')
    parser.add_argument('--action_size', type=int, default=100,
                        help='pattern size to be evaluate')
    parser.add_argument('--depth', type=int, default=5,
                        help='max_searching depth in the MCTS')
    parser.add_argument('--prior_policy', default='pmsn',
                        help='alternative w2v:word2vector')
    parser.add_argument('--quick_policy', default='rlof',
                        help='alternative pmsn')
    parser.add_argument('--entity_encoding', default="pmsn",
                        help='alternative w2v:word2vector')
    parser.add_argument('--context_count', type=int, default=100,
                        help='context_pattern size used in the PMSN')
    parser.add_argument('--device', type=int, default=-1,
                        help='cuda device to be used')
    parser.add_argument('--seed', type=int, default=None,
                        help='manual seed')
    parser.add_argument('--no_search', action='store_true',
                        help='disable the MCTS method')
    parser.add_argument('--only_top', action='store_true',
                        help='print only top results')
    return parser


def get_agent(opt):
    params = AgentParams(opt)
    if opt['no_search']:
        agent_class = OneAgent
    else:
        agent_class = MCAgent
    agent = agent_class(params, BootEnv, DomainSemanticModel)
    return agent, params


if __name__ == '__main__':
    opt = vars(regist_parser().parse_args())
    for key, value in opt.items():
        print(key, ':\t', value)
    if opt['seed']:
        np.random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
    agent, params = get_agent(opt)

    state = agent.test_model()
    for entity, _ in state.get_top_entities():
        print('%s' % (entity))
