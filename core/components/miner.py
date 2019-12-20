# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-09-29 15:14:09
Last modified: 2019-01-11 12:44:22
Python release: 3.6
Notes:
"""
import sqlite3
import logging
from collections import Iterable
import os

import nltk
from utils.net_io import NetLink
from yan_tools.io.data_io import load_data
from core.components.pattern_process import filter_ngrams
from core.components.pattern_process import filter_entity_func
from core.components.pattern_process import filter_pattern_func

CNT = 0
NG = CNT + 1
PRB = NG + 1
ABS = PRB + 1
KN = ABS + 1
KNMC = KN + 1
DIR = KNMC + 1
DKN = DIR + 1

ENTITY = 0
PATTERN = 1
table_names = ['Entity', 'Pattern']


class Cache(object):
    """数据持久层
    """

    def __init__(self, file_name, logger, update):
        self.file_name = file_name
        self.entity_queries = {}
        self.pattern_queries = {}
        self.e_new = []
        self.p_new = []
        self.logger = logger
        self.update = update
        self.load()

    def add(self, key, value, dtype=ENTITY):
        """

        :param key:
        :param value:

        """
        queries, new = self._get_query_table(dtype)
        if key not in queries:
            queries[key] = value
            if self.update:
                new.append((key, value))
                check_time = 50 if dtype == ENTITY else 10000
                if len(new) >= check_time:
                    self._save(new, dtype)

    def _save(self, new, dtype=ENTITY):
        table_name = table_names[dtype]
        if len(new) > 0:
            with sqlite3.connect(self.file_name) as conn:
                cursor = conn.cursor()
                temp_values = []
                for key, values in new:
                    temp_value = []
                    for v_str, count in values:
                        temp_value.append((v_str + ' ' + str(count)))
                    temp_values.append((key, '|'.join(temp_value)))
                cursor.executemany('insert into %s values(?, ?)' %
                                   (table_name), temp_values)

            self.logger.info('saved %d for %s' % (len(new), table_name))
            if dtype == ENTITY:
                self.e_new = []
            else:
                self.p_new = []

    def _get_query_table(self, dtype):
        if dtype == ENTITY:
            return self.entity_queries, self.e_new
        else:
            return self.pattern_queries, self.p_new

    def save(self):
        self._save(self.p_new, PATTERN)
        self._save(self.e_new, ENTITY)

    def load(self):
        if os.path.exists(self.file_name):
            with sqlite3.connect(self.file_name) as conn:
                cursor = conn.cursor()
                self._load(cursor, 'select * from Entity', self.entity_queries)
                self._load(cursor, 'select * from Pattern',
                           self.pattern_queries)
            self.logger.info(
                'loaded ' + str(len(self.entity_queries)) + ' entity queries')
            self.logger.info(
                'loaded ' + str(len(self.pattern_queries)) +
                ' pattern_queries')

    def _load(self, cursor, sql_str, queries):
        cursor.execute(sql_str)
        for query, value in cursor.fetchall():
            tuples = []
            for entry in value.split('|'):
                entry = entry.strip()
                if len(entry):
                    entries = entry.split(' ')
                    tuples.append((' '.join(entries[:-1]), int(entries[-1])))
            queries[query] = tuples

    def get(self, key, dtype=ENTITY):
        """

        :param key:

        """
        queries, _ = self._get_query_table(dtype)
        if key in queries:
            return queries[key]
        return None


class Miner(object):
    """ """

    def __init__(self, host=None, port=None, stop_words=set(),
                 cache_path=None, update=True, pattern_type='middle'):
        super(Miner, self).__init__()
        self.host = host if host else '127.0.0.1'
        self.port = port if port else 13680
        self.stop_words = stop_words
        self.cache = Cache(cache_path, logging.getLogger(), update)
        self.pattern_type = pattern_type
        self.connect()

    def connect(self):
        conn = NetLink(self.host, self.port)
        self.conn = conn.conn

    def close(self):
        if self.conn is not None:
            self.conn.close()

    def match_patterns_by_entity(self, entity_str, method=NG):
        """
        generate match string of a entity
        then match patterns by that string

        :param entity_str:
        :param method:  (Default value = NG)

        """
        return self.match(entity_str,
                          self.entity_preprocess,
                          self.entity_postprocess,
                          ENTITY)

    def match_entities_by_pattern(self, pattern_str):
        """
        generate match string of a pattern
        then match entities by that string

        :param pattern_str:
        :param method:  (Default value = NG)

        """
        return self.match(pattern_str,
                          self.pattern_preprocess,
                          self.pattern_postprocess,
                          PATTERN)

    def match(self, string, preprocess, postprocess, dtype):
        ret_data = self.cache.get(string, dtype)
        if ret_data is None:
            ret_data = []
            match_patterns, positions = preprocess(string)
            for pattern, position in zip(match_patterns, positions):
                matched = postprocess(self._match(pattern), position)
                ret_data.extend(matched)
            self.cache.add(string, ret_data, dtype)
        return ret_data

    def _match(self, match_str):
        """
        :param query:
        :param method:  (Default value = NG)

        """
        request_code = self.encode(match_str)
        self.conn.sendall(request_code)
        matched_lines = self.decode(self.conn)
        return matched_lines

    @staticmethod
    def decode(conn):
        """

        :param ret_string:
        :param method:  (Default value = NG)

        """
        response = conn.makefile()
        first_line = response.readline()
        length = int(first_line.strip())
        lines = []
        for i in range(length):
            lines.append(response.readline().strip())
        assert length == len(lines)
        return lines

    @staticmethod
    def encode(req_string):
        """

        :param match_string:
        :param method:  (Default value = NG)

        """
        method = NG
        req_string = 'english ' + req_string
        b_head = bytes(str(method), encoding='utf-8')
        s_length = str(len(req_string)).zfill(4)
        b_len = bytes(s_length, encoding='utf-8')
        b_query = bytes(req_string, encoding='utf-8')
        b_request = b_head + b_len + b_query
        return b_request

    def entity_preprocess(self, entity_str):
        """

        :param entity_str:
        :param ranges:

        """
        patterns = []
        positions = []
        entity_split = entity_str.split(' ')
        entity_len = len(entity_split)
        if entity_len < 2:
            if self.pattern_type == 'middle':
                pattern = ['_'] * 2 + entity_split + ['_'] * 2
                patterns.append(' '.join(pattern))
                positions.append(2)
            else:
                ins_first = entity_split + ['_'] * 4
                ins_last = ['_'] * 4 + entity_split
                patterns.extend([' '.join(ins_first), ' '.join(ins_last)])
                positions.extend([0, 4])
        return patterns, positions

    def entity_postprocess(self, lines, position):
        """

        :param string:
        :param positions:

        """
        return filter_ngrams(lines, position,
                             filter_pattern_func, self.stop_words)

    def pattern_preprocess(self, pattern, max_len=5):
        """

        :param pattern:
        :param max_len:  (Default value = 5)

        """
        pattern_split = pattern.strip().split(' ')
        pattern_len = len(pattern_split)
        if pattern_len != 5:
            raise ValueError('invalid pattern')

        if pattern_split[0].strip() == '_':
            start = 0
        elif pattern_split[4].strip() == '_':
            start = 4
        elif pattern_split[2].strip() == '_':
            start = 2
        else:
            raise ValueError('invalid pattern')
        patterns = [pattern.strip()]
        positions = [start]
        return patterns, positions

    def pattern_postprocess(self, lines, position):
        """

        :param string:
        :param position:

        """
        return filter_ngrams(lines, position,
                             filter_entity_func, self.stop_words)


if __name__ == '__main__':
    stop_words = set(nltk.corpus.stopwords.words('english'))
    STOP_WORDS_FILE = './data/stopwords.txt'
    stop_words.update(load_data(STOP_WORDS_FILE))
    miner = Miner(host='192.168.0.109', stop_words=stop_words,
                  cache_path='./caches/cache.db')
    results = miner.match_entities_by_pattern('_ is the capital of')
    if isinstance(results, Iterable):
        for r in results:
            print(r[0] + ' ' + str(r[1]))
        print(len(results))
    else:
        print(results)
