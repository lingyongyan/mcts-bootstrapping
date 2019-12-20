# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-15 18:14:36
Last modified: 2019-02-21 18:49:17
Python release: 3.6
Notes:
"""
import logging
from torchtext.data import Field, Dataset, Example


def remove_wild_char(word_list):
    return [w for w in word_list if w != '_']


def avoid_stop_words(x):
    pass


class PatternField(Field):
    def __init__(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to "
                           "use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set "
                           "to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True
        if kwargs.get('lower') is True:
            logger.warning("Option lower has to be set to use "
                           "pretrained embedding.  Changed to False.")
        kwargs['lower'] = True
        kwargs['preprocessing'] = remove_wild_char

        super(PatternField, self).__init__(*args, **kwargs)

    def build_vocab(self, word_vectors=None, itos=None):
        assert word_vectors is not None or itos is not None
        if word_vectors is not None:
            words = [word_vectors.itos]
            super(PatternField, self).build_vocab(words, vectors=word_vectors)
        else:
            words = [itos]
            super(PatternField, self).build_vocab(words)


class PatternDataset(Dataset):
    def __init__(self, pattern_list, fields, **kwargs):
        examples = []
        for pattern in pattern_list:
            if not isinstance(pattern, str):
                pattern = str(pattern)
            example = Example.fromlist([pattern], fields)
            examples.append(example)
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        super(PatternDataset, self).__init__(examples, fields, **kwargs)
