# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-18 17:40:44
Last modified: 2019-05-20 08:47:07
Python release: 3.6
Notes:
"""
from numbers import Number
from sklearn.cluster import MiniBatchKMeans
import torch
import heapq
import numpy as np
from core.agents.basic_agent import BasicAgent
from core.utils.torchutil import weight_normalize
from core.components.filters import sort_semantic_patterns
from core.components.filters import sort_core_patterns_by_count
from core.models.similarity_measure import pms_fast, pms_relax
from core.models.similarity_measure import cosine_similarity
from core.models.similarity_measure import parallel_batch_pms_fast
from core.models.similarity_measure import parallel_batch_pms_relax
from core.models.similarity_measure import parallel_batch_pms_relax_check
from core.models.similarity_measure import parallel_batch_word_mover_distance


class autovivify_list(dict):
    '''Pickleable class replicate functionality of collections.defaultdict'''

    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


class NetAgent(BasicAgent):
    def __init__(self, args, env_prototype, model_prototype,
                 memory_prototype=None):
        super(NetAgent, self).__init__(args, env_prototype,
                                       model_prototype, memory_prototype)

    def state_predict(self, state):
        if not self.env.is_ended(state):
            pi, _ = self.model.predict(state, state.actions)
            return pi
        else:
            valid_actions = state.actions
            ones = np.ones((len(valid_actions),))
            ones /= ones.sum()
            return ones.tolist()

    def pattern_sim(self, pattern1, pattern2, sim_type):
        pattern1 = [pattern1]
        pattern2 = [pattern2]
        if sim_type == 'pattern_embedding':
            p1_emb = self.env.reward_model.encoder._embedding(pattern1)
            p2_emb = self.env.reward_model.encoder._embedding(pattern2)
            scores = cosine_similarity(p1_emb, p2_emb)
        score = scores.view(-1).cpu().numpy().tolist()[0]
        return score

    def core_patterns(self, entities, num, _type):
        if _type == 'emb':
            left_patterns = []
            right_patterns = []
            all_num = len(entities)
            link_patterns = self.env.storage.get_normal_patterns_by_entities(
                entities, 0.1)
            patterns = [p for p, counts in link_patterns.items()]
            e1_embs = self.env.reward_model.encoder._embedding(entities)
            e2_embs = self.env.reward_model.encoder._embedding(patterns)
            direct_scores = cosine_similarity(e1_embs, e2_embs).mean(
                dim=0).cpu().numpy().tolist()

            for (p, counts), weight in \
                    zip(link_patterns.items(), direct_scores):
                match = len(counts)
                link_es = self.env.storage.pattern_pool.links[p]
                df = len(link_es)
                beaf = df - match
                if (all_num > 1 and match < 2) or beaf < 5 or beaf > 2000:
                    continue
                if p[0] == '_':
                    right_patterns.append((p, match, df, weight))
                else:
                    left_patterns.append((p, match, df, weight))
            if num > 0:
                left_patterns = heapq.nlargest(num, left_patterns,
                                               key=lambda x: x[-1])
                right_patterns = heapq.nlargest(
                    num, right_patterns, key=lambda x: x[-1])
            results = left_patterns + right_patterns
        else:
            results = sort_core_patterns_by_count(
                self.env.storage, entities, num=num)
        return results

    def entity_sim(self, entities, entities2, sim_type):
        return [self._entity_sim(entities, entities2, sim_type)]
        '''
        results = []
        for entity in entities2:
            results.append(self._entity_sim(entities, [entity], sim_type))
        return results
        '''

    def _entity_sim(self, entities, entity, sim_type):
        e1_embs = self.env.reward_model.encoder._embedding(entities)
        e2_embs = self.env.reward_model.encoder._embedding(entity)
        e1_embs = torch.mean(e1_embs, dim=0, keepdim=True)
        e2_embs = torch.mean(e2_embs, dim=0, keepdim=True)
        direct_scores = cosine_similarity(e1_embs, e2_embs).mean(
            dim=0)
        if sim_type != 'entity_embedding':
            e1_embs = self.env.reward_model.encoder.encode(
                entities, storage=self.env.storage, type_='entity', num=300)
            e2_embs = self.env.reward_model.encoder.encode(
                entity, storage=self.env.storage, type_='entity', num=300)
            if sim_type == 'pattern_embedding_pms_fast':
                scores = parallel_batch_pms_fast(e1_embs, e2_embs)
            elif sim_type == 'pattern_embedding_pms_relax':
                scores = parallel_batch_pms_relax(e1_embs, e2_embs)
            else:
                scores = parallel_batch_word_mover_distance(e1_embs, e2_embs)
            scores = torch.mean(scores, dim=-1)
        else:
            scores = direct_scores
        direct_scores = direct_scores.view(-1).cpu().numpy().tolist()[0]
        mean_score = scores.mean(dim=-1).view(-1).cpu().numpy().tolist()[0]
        max_score = scores.max(dim=-1)[0].view(-1).cpu().numpy().tolist()[0]
        min_score = scores.min(dim=-1)[0].view(-1).cpu().numpy().tolist()[0]
        return ','.join(entities), ','.join(entity), direct_scores,\
            mean_score, max_score, min_score

    def pms_check(self, entities_1, entities_2):
        p1s, w1s, p1sr, w1sr = \
            self.env.reward_model.encoder.dpe.dsm_init_weight(
                self.env.storage, None, entities_1,
                type_='entity_set', num=100)[0]

        p2s, w2s, p2sr, w2sr = \
            self.env.reward_model.encoder.dpe.dsm_init_weight(
                self.env.storage, None, entities_2,
                type_='entity_set', num=150)[0]

        e1_embs = self.env.reward_model.encoder.encode(
            entities_1, storage=self.env.storage, type_='entity_set', num=100)
        e2_embs = self.env.reward_model.encoder.encode(
            entities_2, storage=self.env.storage, type_='entity_set', num=150)
        left, right = parallel_batch_pms_relax_check(e1_embs, e2_embs)
        left = left.squeeze(0).squeeze(0)
        right = right.squeeze(0).squeeze(0)
        l1_sim = left.max(dim=1)[0].view(-1).cpu().numpy().tolist()
        l1_indices = left.max(dim=1)[1].view(-1).cpu().numpy().tolist()
        l1_r_sim = right.max(dim=1)[0].view(-1).cpu().numpy().tolist()
        l1_r_indices = right.max(dim=1)[1].view(-1).cpu().numpy().tolist()
        result = []
        for i in range(len(p1s)):
            result.append((w1s[i], p1s[i], l1_sim[i],
                           p2s[l1_indices[i]], w2s[l1_indices[i]]))
        for i in range(len(p1sr)):
            result.append((w1sr[i], p1sr[i], l1_r_sim[i],
                           p2sr[l1_r_indices[i]], w2sr[l1_r_indices[i]]))
        return result

    def emb(self, entities):
        left_patterns, right_patterns = sort_semantic_patterns(
            self.env.storage, entities)
        left_patterns, left_weights = left_patterns
        right_patterns, right_weights = right_patterns
        left_embs = self.env.reward_model.encoder._embedding(
            left_patterns).cpu().numpy()
        right_embs = self.env.reward_model.encoder._embedding(
            right_patterns).cpu().numpy()
        return left_embs, left_weights, right_embs, right_weights

    def cluster(self, entities):
        left_embs, left_weights, right_embs, right_weights = self.emb(entities)
        kmeans_model = MiniBatchKMeans(init='k-means++', n_clusters=100)
        kmeans_model.fit(left_embs, sample_weight=left_weights)
        kmeans_model2 = MiniBatchKMeans(init='k-means++', n_clusters=100)
        kmeans_model2.fit(right_embs, sample_weight=right_weights)
        return kmeans_model, kmeans_model2, left_weights, right_weights

    def cluster_emb(self, entities):
        kmeans_model, kmeans_model2, left_weights, right_weights =\
            self.cluster(self, entities)
        left_cluster_labels = kmeans_model.labels_
        right_cluster_labels = kmeans_model2.labels_
        left_emb = kmeans_model.cluster_centers_
        right_emb = kmeans_model2.cluster_centers_
        left_weight_result = autovivify_list()
        right_weight_result = autovivify_list()
        for w, c in zip(left_weights, left_cluster_labels):
            left_weight_result[c].append(w)
        for w, c in zip(right_weights, right_cluster_labels):
            right_weight_result[c].append(w)
        left_weight = [0.] * 100
        right_weight = [0.] * 100
        for c, weights in left_weight_result.items():
            left_weight[c] = sum(weights)
        for c, weights in right_weight_result.items():
            right_weight[c] = sum(weights)
        left_emb = torch.tensor(left_emb, dtype=torch.float).unsqueeze(0)
        left_weight = weight_normalize(left_weight, min_v=1e-3, max_v=0.2)
        left_weight = torch.tensor(left_weight, dtype=torch.float).unsqueeze(0)
        right_emb = torch.tensor(right_emb, dtype=torch.float).unsqueeze(0)
        right_weight = weight_normalize(right_weight, min_v=1e-3, max_v=0.2)
        right_weight = torch.tensor(
            right_weight, dtype=torch.float).unsqueeze(0)
        return left_emb, left_weight, right_emb, right_weight
