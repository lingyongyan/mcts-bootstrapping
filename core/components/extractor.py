# coding=utf-8
""""
Program:
Description: search the Web 1T n-grams corpus by url access
Author: lingyongy(lingyongy@qq.com)
Date: 2018-07-04 15:28:15
Last modified: 2018-12-22 23:17:43
Python release: 3.6
"""
import logging


class Extractor(object):
    """ 抽取器

    Attributes:
        miner: miner
        entity_pool: pool storing entities
        pattern_pool: pool storing patterns
        pattern_entities: entities extended by some pattern
        lst_new_patterns: last new patterns
        last_new_entities: last new entities
    """

    def __init__(self, miner):
        self.miner = miner
        self.entity_history = set()
        self.pattern_history = set()
        self.logger = logging.getLogger(__name__)

    def match_patterns_by_entities(self, storage, entity_contents):
        """ extend patterns using entity

        :param entity:

        """
        for entity_content in entity_contents:
            if entity_content not in self.entity_history:
                pattern_contents = self.miner.match_patterns_by_entity(
                    entity_content)
                storage.add_matched_patterns_of_entity(
                    entity_content, pattern_contents)
                self.entity_history.add(entity_content)
                self.logger.debug('Get %d patterns for entity "%s"' %
                                  (len(pattern_contents), entity_content))

    def match_entities_by_patterns(self, storage, pattern_contents):
        """

        :param pattern:

        """
        for pattern_content in pattern_contents:
            if pattern_content not in self.pattern_history:
                entity_contents = self.miner.match_entities_by_pattern(
                    pattern_content)
                storage.add_matched_entities_of_pattern(
                    pattern_content, entity_contents)
                self.pattern_history.add(pattern_content)
                self.logger.debug('Get %d entities for pattern "%s"' %
                                  (len(entity_contents), pattern_content))
