# coding=utf-8
""""
Program: Regex config
Description:
Author: lyy
Date: 2018-07-18 16:23:27
Last modified: 2019-01-11 13:04:53
Python release: 3.6
"""
import regex as re
# import spacy
# import nltk
# from nltk.tokenize.toktok import ToktokTokenizer
# from nltk.tag.perceptron import PerceptronTagger

WORD_PATTERN = re.compile(r'\'?[a-zA-Z]+(\-[a-zA-Z]+)?$')
SEN_PATTERN = re.compile(r'[a-zA-Z \'\-]+$')
NUM_PATTERN = re.compile(r'[0-9]+')

# nlp = spacy.load('en', parser=False, entity=False, ner=False)
# toktok = ToktokTokenizer()
# tagger = PerceptronTagger()
'''
neg_pos1 = set(('CD', 'LS', 'SYM'))
neg_pos2 = set(('NN', 'NNS', 'NNP', 'NNPS', 'UH', 'FW', 'CC'))
real_pos1 = set(('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN', 'NNS',
                 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'))
real_pos2 = set(('NN', 'NNS', 'NNP', 'NNPS'))
'''
'''
neg_pos1 = set(('NUM', 'SYM', 'X', 'PUNCT'))
neg_pos2 = set(('NOUN', 'PROPN', 'INTJ', 'CCONJ'))
real_pos1 = set(('ADJ', 'ADV', 'NOUN', 'PRONPN', 'VERB'))
real_pos2 = set(('NOUN', 'PROPN'))
'''


def is_sen(string):
    if SEN_PATTERN.match(string):
        return True
    return False


def filter_stopwords(words, stop_words):
    if isinstance(words, str):
        words = [words]
    filtered_words = [w for w in words if w.lower() not in stop_words
                      and w not in stop_words and len(w) > 2]
    return filtered_words


def filter_ngrams(lines, pos, filter_func, stop_words):
    ret_data = []
    for line in lines:
        rindex = line.rfind(' ')
        left = line[:rindex]
        words = left.split(' ')
        if rindex > 0 and is_sen(left) and len(words) == 5:
            left_part = filter_func(words, pos, stop_words)
            if left_part:
                ret_data.append((left_part, int(line[rindex + 1:])))
    return ret_data


def filter_entity_func(left, pos, stop_words):
    entity = left[pos]
    if len(filter_stopwords(entity, stop_words)) > 0:
        if WORD_PATTERN.match(entity) is not None and entity.istitle():
            return entity
    return None


def check_pattern(pattern, check_pos):
    if not pattern[check_pos][0].isupper():
        return True
    return False


def filter_pattern(pattern, stop_words):
    if len(filter_stopwords(pattern, stop_words)) > 0:
        for w in pattern:
            if WORD_PATTERN.match(w) is None:
                return False
        return True
    return False


def filter_pattern_func(left, pos, stop_words):
    flag = False
    if pos == 2:
        left_pattern = left[:pos]
        right_pattern = left[pos+1:]
        if filter_pattern(left_pattern+right_pattern, stop_words) and\
                check_pattern(left_pattern, -1) and\
                check_pattern(right_pattern, 0):
            flag = True
    elif pos == 0:
        pattern = left[pos+1:]
        if filter_pattern(pattern, stop_words) and check_pattern(pattern, 0):
            flag = True
    else:
        pattern = left[:pos]
        if filter_pattern(pattern, stop_words) and check_pattern(pattern, -1):
            flag = True

    if flag:
        left[pos] = '_'
        pattern = ' '.join(left)
        return pattern
    return None
