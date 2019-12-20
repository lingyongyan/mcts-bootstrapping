# coding=utf-8
""""
Program:
Description:
Author: lingyongy(lingyongy@qq.com)
Date: 2018-07-02 10:33:43
Last modified: 2019-01-09 15:59:45
Python release: 3.6
"""
import math
import os
import pickle

from yan_tools.io.data_io import load_data


class Externals(object):
    """ """
    def __init__(self,
                 stop_words,
                 golden_set,
                 wordlist=None):
        self.stop_words = stop_words
        self.golden_set = golden_set
        self.wordlist = wordlist


class WordList(object):
    """ """
    def __init__(self, file_path):
        self.word_idf = {}
        self.word_count = {}
        self.all_count = 0
        self._read_wordlist(file_path)

    def _read_wordlist(self, file_path):
        """

        :param file_path:

        """
        pkl_name = file_path + '.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as file_:
                self.all_count, self.word_count, self.word_idf = pickle.load(
                    file_)

        else:
            for words in load_data(file_path, lambda x: x.split('\t')):
                count = int(words[1])
                if count >= 3:  # word frequency should no less than 3
                    self.word_count[words[0]] = count
                    self.all_count += count
            for word in self.word_count:
                self.word_idf[word] = math.log(
                    self.all_count / (1 + self.word_count.get(word, 0)))
            with open(pkl_name, 'wb') as filew:
                pickle.dump(
                    (self.all_count, self.word_count, self.word_idf), filew)

    def get_idf(self, word):
        """

        :param word:

        """
        return self.word_idf.get(word, 0)


