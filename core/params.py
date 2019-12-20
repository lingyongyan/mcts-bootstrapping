# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-10-17 16:22:05
Last modified: 2019-12-20 09:06:58
Python release: 3.6
Notes:
"""
import os
import math
from collections import defaultdict
from functools import partial
import heapq

import numpy as np
import nltk
import torch
from torchtext.vocab import Vectors

from yan_tools.logger import get_logger
from yan_tools.io.data_io import load_data

from core.components.externals import Externals
from core.components.storage import load_storage
# from core.components.filters import sort_expanding_patterns_by_meb
from core.components.filters import sort_expanding_patterns_by_rlogf
from core.components.filters import get_expanding_patterns
from core.models.encoders.encoder_models import CBOWEncoder
# from core.models.encoders.encoder_models import RNNEncoder
from core.models.encoders.pmsn import PMSN
from core.models.encoder import Encoder
from core.models.policies.semantic_policy import SemanticPolicy
from core.models.policies.rlogf_policy import RlogfPolicy
from core.models.scorers.emb_scorer import EmbScorer
from core.models.similarity_measure import Sim, cosine_similarity
from core.models.similarity_measure import batch_pms_relax
from core.utils.torchtextutil import PatternDataset, PatternField


POS_SEED_FILE = './data/%s_positive_seed.txt'
NEG_SEED_FILE = './data/%s_negative_seed.txt'

STOP_WORDS_FILE = './data/stopwords.txt'

RNN_PARAMS = './caches/rnn.pkl'
CBOW_PARAMS = './caches/cbow.pkl'

stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.update(load_data(STOP_WORDS_FILE))
golden_set = set([])

CONFIGS = [
    ['ours', 'yan', '', 'semantic', 'cbow', 'dsm', 'mean', 'net'],
    ['ours', 'yan', 'rlogf', 'semantic', 'cbow', 'dsm', 'min', 'mcts'],
    ['ours', 'yan', 'rlogf', 'semantic', 'cbow', 'dsm', 'mean', 'mcts'],
    ['ours', 'yan', 'rlogf', 'semantic', 'cbow', 'w2v', 'mean', 'mcts'],
    ['ours', 'yan', 'rlogf', 'semantic', 'cbow', 'dsm', 'mean', 'topn'],
    ['ours', 'yan', 'rlogf', 'semantic', 'cbow', 'w2v', 'mean', 'topn'],
]


def argmax(arr):
    _, _, index = heapq.nlargest(1, arr, key=lambda x: (x[0], x[1]))[0]
    return index


def top_argmax(arr, top_n):
    filterd = [(n, v, k) for n, v, k in arr if n > 2]
    tops = heapq.nlargest(top_n, filterd, key=lambda x: x[1])
    indices = [value[2] for value in tops]
    return indices


def half():
    return 0.3


pos_thresholds = defaultdict(half, [])


class Params(object):   # NOTE: shared across all modules
    def __init__(self, opt):
        self.verbose = 1            # 0(warning) | 1(info) | 2(debug)
        self.root_dir = os.getcwd()
        device_id = opt['device']
        self.device = torch.device("cuda:%d" % device_id if device_id > 0 and
                                   torch.cuda.is_available() else "cpu")


class EnvParams(Params):    # settings for simulation environment
    def __init__(self, opt, logger, encoder, sim_measure):
        super(EnvParams, self).__init__(opt)
        self.logger = logger
        self.should_shuffle = True
        self.externals = Externals(stop_words, golden_set)

        assert os.path.exists(opt['dataset'])
        self.logger.info('Load from %s' % opt['dataset'])
        self.storage = load_storage(opt['dataset'])

        self.max_iters = 25
        self.max_entities = 300
        self.get_expanding_patterns = get_expanding_patterns
        self.sort_expanding_patterns = sort_expanding_patterns_by_rlogf
        # self.exploit_sort = sort_expanding_patterns_by_meb
        self.reward_model = EmbScorer(encoder, sim_measure, logger,
                                      opt['entity_encoding'])
        self.logger.warning("<===========Loaded Env Parameters===========>")


class EncoderParams(Params):
    def __init__(self, opt, logger):
        super(EncoderParams, self).__init__(opt)
        self.context_count = opt['context_count']
        self.batch_size = 512
        self.vocab_size = 1496009
        self.dim = 300
        self.vec_path = './caches'
        self.dataset_cls = PatternDataset
        self.name = 'glove.840B.300d_eng.txt'
        itos_path = './caches/glove_840B_300d_eng.vocab.pt'
        '''
        self.name = 'GoogleNews-vectors-negative300.txt'
        itos_path = './caches/GoogleNews-vectors-negative300.vocab.pt'
        '''
        self.field = PatternField(stop_words=stop_words)

        CBOW_PARAMS = './caches/cbow.pkl'

        logger.info('initialize CBOW encoder')
        if os.path.exists(CBOW_PARAMS):
            logger.info('load itos from cache file %s' % itos_path)
            itos, stoi = torch.load(itos_path)
            self.field.build_vocab(itos=itos)
            self.encoder = CBOWEncoder(
                vocab_size=self.vocab_size, emb_dim=self.dim)
            logger.info('load CBOW encoder parameters from %s' %
                        CBOW_PARAMS)
            self.encoder.load_state_dict(torch.load(CBOW_PARAMS))
        else:
            vectors = Vectors(self.name, self.vec_path)
            torch.save((vectors.itos, vectors.stoi), itos_path)
            self.field.build_vocab(word_vectors=vectors)
            self.encoder = CBOWEncoder(
                vocab_size=self.vocab_size,
                vectors=self.field.vocab.vectors)
            logger.info('save CBOW encoder parameters to%s' %
                        CBOW_PARAMS)
            torch.save(self.encoder.state_dict(), CBOW_PARAMS)
        self.encoder.to(self.device)
        self.logger = logger
        self.logger.warning("<===========Loaded Encoder Parameters==========>")


class ModelParams(Params):  # settings for network architecture
    def __init__(self, opt, logger, encoder, sim_measure):
        super(ModelParams, self).__init__(opt)
        if opt['prior_policy'] == 'rlogf':
            self.policy = RlogfPolicy(logger)
        elif opt['prior_policy'] == 'pmsn':
            self.policy = SemanticPolicy(encoder, sim_measure, logger)
        else:
            raise NotImplementedError()
        self.logger = logger
        self.logger.warning("<===========Loaded Model Parameters============>")


class MCTSParams(object):
    def __init__(self, opt, logger):
        # we use the cpus as mean_reward * √(n_sims)

        self.cpuct = 0.3 * math.sqrt(2 * opt['n_simulations'])

        self.gamma = 0.75
        self._lambda = 1
        self.n_entities = opt['n_entities']
        self.sim_num = opt['n_simulations']
        self.max_depth = math.ceil(opt['depth'] / 2)
        self.total_depth = opt['depth']
        self.logger = logger


class RolloutParams(object):
    def __init__(self, opt, logger):
        self.gamma = 0.75
        self.logger = logger
        self.n_entities = opt['n_entities']
        self.policy = RlogfPolicy(self.logger)


class AgentParams(Params):  # hyperparameters for drl agents
    def __init__(self, opt):
        super(AgentParams, self).__init__(opt)

        self.logger = get_logger(None, self.verbose)

        print_infos = [str(key)+'\t:'+str(value) for key, value in opt.items()]
        self.logger.info(print_infos)
        self.ps = list(load_data(POS_SEED_FILE % opt['entity_type'],
                                 dealer=lambda x: x.strip()))
        self.ns = list(load_data(NEG_SEED_FILE % opt['entity_type'],
                                 dealer=lambda x: x.strip()))
        self.pos_threshold = pos_thresholds[opt['entity_type']]
        self.depth = opt['depth']
        self.action_size = opt['action_size']
        self.n_entities = opt['n_entities']
        self.n_patterns = opt['n_patterns']
        self.only_top = opt['only_top']

        agent_type = 'topn' if opt['no_search'] else 'mcts'

        path = '_'.join([opt['entity_type'], agent_type, opt['prior_policy'],
                         opt['quick_policy'], opt['entity_encoding'],
                         'context%d' % opt['context_count'],
                         'depth%d' % opt['depth']])
        self.cache_path = os.path.join('./results', path)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if agent_type == 'topn':
            self.select_strategy = lambda x, y: np.argpartition(x, -1)[-y:]
        else:
            self.select_strategy = top_argmax
            self.mcts_args = MCTSParams(opt, self.logger)
            self.rollout_args = RolloutParams(opt, self.logger)

        encoder_params = EncoderParams(opt, self.logger)
        sim_dim = -1 if opt['entity_encoding'] == 'pmsn' else 0
        final_sim = partial(torch.mean, dim=sim_dim, keepdim=True)

        if opt['entity_encoding'] == 'pmsn':
            encoder = PMSN(encoder_params)
            sim_measure = Sim(batch_pms_relax, final_sim, cached=False)
            p_encoder = encoder
            policy_sim_measure = Sim(batch_pms_relax, final_sim,
                                     cached=False)
        else:
            encoder = Encoder(encoder_params)
            sim_measure = Sim(cosine_similarity, final_sim, cached=False)
            p_encoder = Encoder(encoder_params)
            policy_sim_measure = Sim(cosine_similarity, final_sim)

        self.model_params = ModelParams(opt, self.logger, p_encoder,
                                        policy_sim_measure)

        self.env_params = EnvParams(opt, self.logger, encoder, sim_measure)
        self.logger.warning("<===========Loaded Agent Parameters============>")
