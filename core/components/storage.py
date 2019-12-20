# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-03 18:30:18
Last modified: 2019-08-25 15:13:42
Python release: 3.6
Notes:
"""
import math
import regex as re
import gc
import pickle

from tqdm import tqdm

from yan_tools.io.data_io import load_data
from core.components.item import ItemPool, EntityPool
import logging


class Storage(object):
    """ """

    def __init__(self):
        self.entity_pool = EntityPool()
        self.pattern_pool = ItemPool()
        self.logger = logging.getLogger()

    def add_matched_patterns_of_entity(self, entity, patterns):
        """
        add the link-to patterns of an entity to entity_pool

        :param entity:
        :param pattern_contents:

        """
        self.entity_pool.push(entity, patterns)

    def add_matched_entities_of_pattern(self, pattern, entities):
        """
        add the link-to entities of an pattern to pattern_pool

        :param pattern:
        :param entity_contents:

        """
        self.pattern_pool.push(pattern, entities)

    def get_entities_by_patterns(self, patterns):
        """

        :param state:
        :param pattern:

        """
        entities = {}
        for p in patterns:
            for e, count in self.pattern_pool.links[p].items():
                if e not in entities:
                    entities[e] = [count]
                else:
                    entities[e].append(count)
        return entities

    def get_patterns_by_entities(self, entities):
        patterns = {}
        for e in entities:
            for p, count in self.entity_pool.links[e].items():
                if p not in patterns:
                    patterns[p] = [count]
                else:
                    patterns[p].append(count)
        return patterns

    def get_normal_patterns_by_entities(self, entities, temperature=0.1):
        patterns = {}
        for e in entities:
            for p, count in self.entity_pool.links[e].items():
                if p not in patterns:
                    patterns[p] = [math.pow(count, temperature)]
                else:
                    patterns[p].append(math.pow(count, temperature))
        return patterns

    @staticmethod
    def load_from_path(*paths, flag=0):
        def read_dealer(line):
            splits = re.split(r'[\t]', line.strip())
            return splits
        entities = {}
        patterns = {}
        for path in paths:
            print('load %s' % path)
            datas = load_data(path, dealer=read_dealer)
            for data in tqdm(datas):
                entity = data[0]
                pattern = data[1]
                count = int(data[2])
                if entity not in entities:
                    entities[entity] = []
                if pattern not in patterns:
                    patterns[pattern] = []
                entities[entity].append((pattern, count))
                patterns[pattern].append((entity, count))
            del datas
        gc.collect()
        storage = Storage()
        for entity, tos in entities.items():
            storage.add_matched_patterns_of_entity(entity, tos)
        for pattern, ptos in patterns.items():
            storage.add_matched_entities_of_pattern(pattern, ptos)
        return storage


def out_filter(entities, patterns):
    removed_patterns = set(
        [p for p, ents in patterns.items() if len(ents) < 2])
    removed_entities = set()
    for entity, pts in entities:
        n_e = [(pt, count) for pt, count in pts if pt not in removed_patterns]
        entities[entity] = n_e
        if len(n_e) == 0:
            removed_entities.add(entity)
    for p in removed_patterns:
        del patterns[p]
    for e in removed_entities:
        del entities[e]
    return entities, patterns


def load_storage(path):
    storage = Storage()
    with open(path, 'rb') as f:
        entity_pool, pattern_pool = pickle.load(f)
    storage.entity_pool = entity_pool
    storage.pattern_pool = pattern_pool
    return storage


def save_storage(storage, path):
    with open(path, 'wb') as f:
        pickle.dump((storage.entity_pool, storage.pattern_pool), f)
