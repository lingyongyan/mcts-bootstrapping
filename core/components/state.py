# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-02 23:33:18
Last modified: 2019-09-01 21:42:06
Python release: 3.6
Notes:
"""
import math
import json
import codecs
import numpy as np
from collections import Counter
import heapq


class State(object):
    """ """

    def __init__(self, pos_seeds, neg_seeds, action_size, pos_threshold,
                 logger=None, only_top=False):
        self.pos_seeds = []
        self.neg_seeds = []
        self.actions = []
        self.priors = []
        self.extracted_patterns = {}
        self.extracted_entities = {}
        self.top = set()
        self.top_added = []
        self.bottom = set()
        self.pos_pattern_counter = Counter()
        self.neg_pattern_counter = Counter()
        self.useless_patterns = set()
        self.used_patterns = set()
        self.pattern_history = []
        self.valid = False
        self.action_size = action_size
        self.feedbacks = {}
        self.adaptions = {}
        self.core_emb = None
        self.only_top = only_top
        self.pos_threshold = pos_threshold
        self.logger = logger

        for entity in pos_seeds:
            if entity.strip():
                self.pos_seeds.append(entity.strip())
        for entity in neg_seeds:
            if entity.strip():
                self.neg_seeds.append(entity.strip())

        self.counter = (0, 0)
        pos_string = '|'.join(set(self.pos_seeds))
        neg_string = '|'.join(set(self.neg_seeds))
        self.base_present = pos_string + '-' + neg_string
        self.string_present = self.base_present

    def expand_entity(self, entity, score, storage, is_top, is_bottom):
        if entity not in storage.entity_pool.links:
            if self.logger:
                self.logger.warn('Not found n_grams for entity:%s!' % entity)
            return
        self.add_entity(entity, score)
        if is_top or is_bottom:
            entity_links = storage.entity_pool.links[entity]
            if is_top:
                self.pos_pattern_counter.update(entity_links.keys())
            else:
                self.neg_pattern_counter.update(entity_links.keys())
            for p, link_times in entity_links.items():
                if p in self.useless_patterns:
                    continue
                if p in self.extracted_patterns:
                    df, pos_count, neg_count = self.extracted_patterns[p]
                    flag = True
                else:
                    pos_count, neg_count = 0, 0
                    df = len(storage.pattern_pool.links[p])
                    flag = False

                pos_match = self.pos_pattern_counter[p]
                neg_match = self.neg_pattern_counter[p]
                if (pos_match + neg_match) > df - 1 or\
                        (pos_match >= 5 and neg_match > pos_match + 1):
                    if flag:
                        del self.extracted_patterns[p]
                    self.useless_patterns.add(p)
                else:
                    if is_top:
                        pos_count += math.pow(link_times, 0.1)
                    else:
                        neg_count += math.pow(link_times, 0.1)
                    self.extracted_patterns[p] = (df, pos_count, neg_count)
            if entity not in self.pos_seeds and entity not in self.neg_seeds:
                if is_top:
                    self.top.add(entity)
                    self.top_added.append(entity)
                else:
                    self.bottom.add(entity)

    def adjust_feedback(self, patterns, weights):
        for p, w in zip(patterns, weights):
            if p not in self.feedbacks:
                self.feedbacks[p] = w
        self.adaptions = {}
        value = sum(list(self.feedbacks.values())) / \
            max(1, len(self.feedbacks))
        value = max(0.01, value)
        if max(weights) > value:
            coordinary = 1 / (max(weights) / value - 1)
        else:
            coordinary = 1
        for p, feedback in self.feedbacks.items():
            self.adaptions[p] = np.clip(1 + coordinary * (feedback / value - 1),
                                        a_min=0.1, a_max=2.0)

    def add_entity(self, entity, weight):
        """

        :param entity:
        :param weight:

        """
        # if entity not in self.pos_seeds and entity not in self.top:
        if entity not in self.pos_seeds and entity not in self.neg_seeds and\
                entity not in self.top and entity not in self.bottom:
            self.extracted_entities[entity] = weight

    def set_entity_weight(self, entity, weight):
        """

        :param entity:
        :param weight:

        """
        if entity in self.extracted_entities:
            self.extracted_entities[entity] = weight

    def get_pattern_tuples(self, patterns):
        """

        :param patterns:

        """
        tuples = []
        for p in patterns:
            if p in self.extracted_pattern:
                tuples.append((p, self.extracted_patterns[p]))
            else:
                tuples.append((p, 0.01))
        return tuples

    def use_patterns(self, patterns):
        """

        :param pattern:

        """
        self.pattern_history.append(patterns)
        for pattern in patterns:
            assert pattern not in self.used_patterns
            del self.extracted_patterns[pattern]
            self.used_patterns.add(pattern)
            self.useless_patterns.add(pattern)
        self.valid = False
        return True

    def get_all_entities(self):
        sorted_entities = list(self.extracted_entities.keys())
        return self.pos_seeds + sorted_entities

    def get_top_entities(self):
        """
        get top extracted entities
        """
        t_entities = [(e, self.extracted_entities[e]) for e in self.top]
        sorted_entities = sorted(t_entities, key=lambda x: x[1], reverse=True)

        # sorted_entities = [e[0] for e in sorted_entities]
        return sorted_entities

    def get_seed_entities(self):
        return self.pos_seeds

    def get_core_entities(self):
        """ """
        sorted_entities = self.get_top_entities()
        core_entities = []
        remain = min(len(self.pattern_history) * 3, 15 - len(self.pos_seeds))
        for entity, weight in sorted_entities:
            if len(core_entities) >= remain or weight < self.pos_threshold:
                break
            core_entities.append(entity)
        return self.pos_seeds + core_entities

    def update_counter(self):
        """ """
        pos, neg = 0, 0
        for entity, weight in self.extracted_entities.items():
            if weight >= self.pos_threshold:
                pos += 1
            else:
                neg += 1
        self.counter = (pos, neg)

    def get_string_representation(self):
        """ """
        self.string_present += self.base_present + \
            '-' + '|'.join(self.used_patterns)
        return self.string_present

    def _to_dict(self):
        if self.only_top:
            # sorted_entities = self.get_top_entities()
            sorted_entities = [(k, self.extracted_entities[k]) for k in
                               self.top_added]
        else:
            all_entities = sorted(self.extracted_entities.items(),
                                  key=lambda x: x[1], reverse=True)
            sorted_entities = [(k, v) for k, v in all_entities if k not in
                               self.bottom and k not in self.neg_seeds]
        state_dict = {'pos_seeds': self.pos_seeds,
                      'neg_seeds': self.neg_seeds,
                      'used_actions': self.pattern_history,
                      'actions': self.actions,
                      'extracted': sorted_entities}
        return state_dict

    def to_json(self):
        """ """
        state_dict = self._to_dict()
        json_dump = json.dumps(state_dict)
        return json_dump

    def to_json_file(self, path):
        state_dict = self._to_dict()
        state_dict['actions'] = state_dict['actions'][:10]
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)

    def update_gate(self):
        count = len(self.pattern_history) * 2
        count = min(count, 10)
        sorted_entities = heapq.nlargest(count,
                                         self.extracted_entities.items(),
                                         key=lambda x: x[1])
        pos_extracted = set()
        for entity, weight in sorted_entities:
            if weight >= self.pos_threshold:
                pos_extracted.add(entity)
        self.pos_extracted = pos_extracted
