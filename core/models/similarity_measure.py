# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-06 11:16:27
Last modified: 2019-08-26 09:28:27
Python release: 3.6
Notes:
"""
from itertools import product
import torch
import pulp


class Sim(object):
    def __init__(self, similarity_measure, final_sim, cached=False):
        self.similarity_measure = similarity_measure
        self.final_sim = final_sim
        self.cache = {}
        self.cached = cached

    def get_sim(self, embeder, core_embs, patterns):
        scores = []
        if self.cached:
            indexer = []
            sub_patterns = []
            for i, p in enumerate(patterns):
                if p in self.cache:
                    scores.append(self.cache[p])
                else:
                    scores.append(0)
                    indexer.append(i)
                    sub_patterns.append(p)
            if len(sub_patterns) > 0:
                entity_embs = embeder(sub_patterns)
                sub_scores = self.similarity_measure(core_embs,
                                                     entity_embs, stric=True)
                sub_scores = self.final_sim(sub_scores)
                sub_scores = sub_scores.view(-1).cpu().numpy().tolist()
                assert len(sub_scores) == len(sub_patterns)
                for i, p, score in zip(indexer, sub_patterns, sub_scores):
                    scores[i] = score
                    self.cache[p] = score
        else:
            entity_embs = embeder(patterns)
            sub_scores = self.similarity_measure(core_embs, entity_embs)
            if sub_scores.dim() > 1:
                sub_scores = self.final_sim(sub_scores)
            scores = sub_scores.view(-1).cpu().numpy()
            assert len(scores) == len(patterns)
        return scores


def cosine_similarity(a, b, dim=-1, eps=1e-8, **kwargs):
    a_norm = a / (a.norm(dim=dim)[:, None]).clamp(min=eps)
    b_norm = b / (b.norm(dim=dim)[:, None]).clamp(min=eps)
    return torch.mm(a_norm, b_norm.transpose(1, 0))


def batch_cosine_similarity(a, b, dim=-1, eps=1e-8, **kwargs):
    a_norm = a / a.norm(dim=dim, keepdim=True).clamp(min=eps)
    b_norm = b / b.norm(dim=dim, keepdim=True).clamp(min=eps)
    sim_matrix = torch.einsum('ijk,lmk->iljm', (a_norm, b_norm))
    return sim_matrix


def pms_relax(a, b, dim=1, eps=1e-8):
    a, a_weights = a
    b, b_weights = b
    a_weights = a_weights.unsqueeze(0)
    b_weights = b_weights.unsqueeze(0)
    a_norm = a / (a.norm(dim=dim)[:, None]).clamp(min=eps)
    b_norm = b / (b.norm(dim=dim)[:, None]).clamp(min=eps)
    sim_matrix = torch.mm(a_norm, b_norm.transpose(1, 0))
    sim_matrix[sim_matrix < 0.3] = 0.
    sim_a = torch.mm(a_weights, sim_matrix.max(dim=1, keepdim=True)[0])
    # sim_b = torch.mm(sim_matrix.max(dim=0, keepdim=True)[0],
    #                  b_weights.transpose(0, 1))
    # sim = torch.min(sim_a, sim_b)
    return sim_a


def batch_pms_relax(a, b, dim=-1, eps=1e-8, stric=False):
    a, a_weights = a
    b, b_weights = b
    a_norm = a / a.norm(dim=dim, keepdim=True).clamp(min=eps)
    b_norm = b / b.norm(dim=dim, keepdim=True).clamp(min=eps)
    sim_matrix = torch.einsum('ijk,lmk->iljm', (a_norm, b_norm)).max(dim=-1)[0]
    if stric:
        sim_matrix[sim_matrix < 0.3] = 0.
    else:
        sim_matrix[sim_matrix < 0.3] = 0.

    sim_a = torch.einsum(
        'ij,ikj->ik', (a_weights, sim_matrix))
    '''
    sim_b = torch.einsum(
        'ijk,jk->ij', (sim_matrix.max(dim=2)[0], b_weights))
    sim = torch.min(sim_a, sim_b)
    '''
    sim = torch.mean(sim_a, dim=0)
    return sim


def parallel_pms_relax(a, b, dim=-1, eps=1e-8):
    l_a, la_w, r_a, ra_w = a
    l_b, lb_w, r_b, rb_w = b
    l_sim = pms_relax((l_a, la_w), (l_b, lb_w))
    r_sim = pms_relax((r_a, ra_w), (r_b, rb_w))
    sim = torch.stack((l_sim, r_sim), dim=-1)
    return sim


def list_parallel_pms_relax(a, bs, dim=-1, eps=1e-8):
    sim_list = []
    for lp, lw, rp, rw in zip(*bs):
        sim_list.append(parallel_pms_relax(a, (lp, lw, rp, rw)))
    return torch.stack(sim_list, dim=0)


def parallel_batch_pms_relax(a, b, dim=-1, eps=1e-8, stric=False):
    l_a, la_w, r_a, ra_w = a
    l_b, lb_w, r_b, rb_w = b
    l_sim = batch_pms_relax((l_a, la_w), (l_b, lb_w), stric=stric)
    r_sim = batch_pms_relax((r_a, ra_w), (r_b, rb_w), stric=stric)
    sim = torch.stack((l_sim, r_sim), dim=-1)
    return sim


def batch_pms_relax_check(a, b, dim=-1, eps=1e-8):
    a, a_weights = a
    b, b_weights = b
    a_norm = a / a.norm(dim=dim, keepdim=True).clamp(min=eps)
    b_norm = b / b.norm(dim=dim, keepdim=True).clamp(min=eps)
    sim_matrix = torch.einsum('ijk,lmk->iljm', (a_norm, b_norm))
    return sim_matrix


def parallel_batch_pms_relax_check(a, b, dim=-1, eps=1e-8):
    l_a, la_w, r_a, ra_w = a
    l_b, lb_w, r_b, rb_w = b
    l_sim_matrix = batch_pms_relax_check((l_a, la_w), (l_b, lb_w))
    r_sim_matrix = batch_pms_relax_check((r_a, ra_w), (r_b, rb_w))
    return l_sim_matrix, r_sim_matrix


def pms_fast(a, b, dim=1, eps=1e-8):
    a, a_weights = a
    b, b_weights = b
    a_weights = a_weights.unsqueeze(0)
    b_weights = b_weights.unsqueeze(0)
    a = torch.mm(a_weights, a)
    b = torch.mm(b_weights, b)
    a_norm = a / (a.norm(dim=dim)[:, None]).clamp(min=eps)
    b_norm = b / (b.norm(dim=dim)[:, None]).clamp(min=eps)
    sim_matrix = torch.mm(a_norm, b_norm.transpose(1, 0))
    return sim_matrix


def batch_pms_fast(a, b, dim=-1, eps=1e-8):
    a, a_weights = a
    b, b_weights = b
    a = torch.einsum('ij,ijk->ik', (a_weights, a))
    b = torch.einsum('ij,ijk->ik', (b_weights, b))
    a_norm = a / a.norm(dim=dim, keepdim=True).clamp(min=eps)
    b_norm = b / b.norm(dim=dim, keepdim=True).clamp(min=eps)
    sim_matrix = torch.mm(a_norm, b_norm.transpose(1, 0))
    return sim_matrix


def parallel_batch_pms_fast(a, b, dim=-1, eps=1e-8):
    l_a, la_w, r_a, ra_w = a
    l_b, lb_w, r_b, rb_w = b
    l_sim = batch_pms_fast((l_a, la_w), (l_b, lb_w))
    r_sim = batch_pms_fast((r_a, ra_w), (r_b, rb_w))
    sim = torch.stack((l_sim, r_sim), dim=-1)
    return sim

# use PuLP


def word_mover_distance_probspec(emb1, emb2, lpFile=None):
    emb1, weights1 = emb1
    emb2, weights2 = emb2
    size_1 = weights1.size(0)
    size_2 = weights2.size(0)
    T = pulp.LpVariable.dicts('T_matrix', list(
        product(range(size_1), range(size_2))), lowBound=0)
    sim_matrix = cosine_similarity(emb1, emb2)
    sim_matrix[sim_matrix < 0.3] = 0.
    prob = pulp.LpProblem('WMD', sense=pulp.LpMaximize)
    # prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[k1, k2] * sim_matrix[k1][k2].item()
                        for k1, k2 in product(range(size_1), range(size_2))])
    for k2 in range(size_2):
        prob += pulp.lpSum([T[k1, k2]
                            for k1 in range(size_1)]) == weights2[k2].item()
    for k1 in range(size_1):
        prob += pulp.lpSum([T[k1, k2]
                            for k2 in range(size_2)]) == weights1[k1].item()

    if lpFile is not None:
        prob.writeLP(lpFile)
    prob.solve()
    return prob


def word_mover_distance(emb1, emb2, lpFile=None):
    prob = word_mover_distance_probspec(emb1, emb2, lpFile=lpFile)
    return pulp.value(prob.objective)


def batch_word_mover_distance(emb1, emb2, lpFile=None):
    emb1, w1 = emb1
    emb2, w2 = emb2
    size1 = emb1.size(0)
    size2 = emb2.size(0)
    values = []
    # sim_matrix = batch_cosine_similarity(emb1, emb2)
    for a, b in product(range(size1), range(size2)):
        prob = word_mover_distance_probspec(
            (emb1[a], w1[a]), (emb2[b], w2[b]), lpFile=lpFile)
        value = pulp.value(prob.objective)
        if value:
            values.append(value)
        else:
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float).view(size1, size2)


def parallel_word_mover_distance(emb1, emb2):
    l_a, la_w, r_a, ra_w = emb1
    l_b, lb_w, r_b, rb_w = emb2
    l_sim = word_mover_distance((l_a, la_w), (l_b, lb_w))
    r_sim = word_mover_distance((r_a, ra_w), (r_b, rb_w))
    return torch.tensor([l_sim, r_sim], dtype=torch.float)


def parallel_batch_word_mover_distance(emb1, emb2):
    l_a, la_w, r_a, ra_w = emb1
    l_b, lb_w, r_b, rb_w = emb2
    l_sim = batch_word_mover_distance((l_a, la_w), (l_b, lb_w))
    r_sim = batch_word_mover_distance((r_a, ra_w), (r_b, rb_w))
    return torch.stack([l_sim, r_sim], dim=-1)


if __name__ == '__main__':
    a = torch.tensor([[[0.7, 0.3], [0.9, 0.1]]])
    a = a.repeat(1, 50, 150)
    a_weights = torch.tensor([[0.01, 0.01]])
    a_weights = a_weights.repeat(1, 50)
    b = torch.tensor([[[0.2, 0.8], [0.7, 0.3]]])
    b = b.repeat(15, 50, 150)
    b_weights = torch.tensor([[0.012, 0.008]])
    b_weights = b_weights.repeat(15, 50)
    import time
    sim2 = batch_pms_relax
    s_time = time.time()
    a = a.cuda()
    a_weights = a_weights.cuda()
    b = b.cuda()
    b_weights = b_weights.cuda()
    for i in range(100):
        distance = sim2((a, a_weights), (b, b_weights))
    print(distance)
    print('time is %.5f s' % (time.time() - s_time))
