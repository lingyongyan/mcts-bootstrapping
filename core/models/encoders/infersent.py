# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-14 15:47:14
Last modified: 2018-12-19 21:42:39
Python release: 3.6
Notes:
"""
import os
import pickle

import torch
from core.models.encoder import Encoder
from core.utils.torchutil import weight_normalize
from utils.InferSent.models import InferSent


class InferSentEncoder(Encoder):
    def __init__(self, args):
        super(InferSentEncoder, self).__init__(args)
        self.infersent = InferSent(args.encoder_params)
        self.ckpt_path = args.ckpt_path
        self.K = args.K
        self.cache_path = args.cache_path
        self.cache = {}
        self.load()
        self.load_model()

    def load_model(self):
        self.infersent.load_state_dict(
            torch.load(self.ckpt_path))
        self.infersent = self.infersent.cuda()
        self.infersent.set_w2v_path(self.vector_path)
        self.infersent.build_vocab_k_words(K=self.K)

    def save(self):
        if self.cache_path:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)

    def load(self):
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)

    def _get_embeddings(self, patterns):
        patterns = [pattern.lower() for pattern in patterns]
        pattern_embeddings = self.infersent.encode(patterns, bsize=256,
                                                   tokenize=False)
        for p, p_emb in zip(patterns, pattern_embeddings):
            self.cache[p] = p_emb

    def get_direct_embeddings(self, patterns):
        new_patterns = [p for p in patterns if p not in self.cache]
        if new_patterns:
            self._get_embeddings(new_patterns)
        return torch.tensor([self.cache[p.lower()] for p in patterns],
                            dtype=torch.float)

    def get_embeddings(self, storage, state, entities):
        entity_embeddings = []
        for entity in entities:
            represent_patterns = storage.get_core_patterns([entity])
            represent_patterns = state.get_pattern_tuples(represent_patterns)
            entity_embeddings.append(
                self.get_patternset_embedding(represent_patterns))
        return torch.stack(entity_embeddings)

    def get_patternset_embedding(self, patterns_weights):
        patterns = [p[0] for p in patterns_weights]
        weights = [p[1] for p in patterns_weights]
        weights = weight_normalize(weights)
        weights = torch.from_numpy(weights).unsqueeze(0).float()
        embeddings = self.get_direct_embeddings(patterns)
        entity_embedding = torch.mm(weights, embeddings)
        return entity_embedding.squeeze(0)

    def encode(self, patterns, storage=None, state=None, type_='pattern'):
        if type_ == 'pattern':
            return self.get_direct_embeddings(patterns)
        elif type_ == 'entity':
            return self.get_embeddings(storage, state, patterns)
        else:
            raise ValueError('type_ should be in ("pattern", "entity")')
