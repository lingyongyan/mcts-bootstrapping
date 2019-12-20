# coding=utf-8
""""
Program: Item
Description: Item means the sematic item
Author: lingyongy(lingyongy@qq.com)
Date: 2018-07-06 10:46:33
Last modified: 2019-02-23 16:29:59
Python release: 3.6
"""


def is_legal(pattern):
    splits = pattern.split(' ')
    if splits[0] == '_':
        return 1
    elif splits[-1] == '_':
        return -1


class Item(object):
    """ Item

    Attributes:
        content: content of item
        weight: confidence of this item
        links: connection weight with other items
        pre_links: previous item link to this item
        post_links: post item link to this item
    """

    def __init__(self, content, weight=None):
        self.content = content
        self.weight = weight

    def __str__(self):
        return self.content


class ItemPool(object):
    """ Item pool storing items.

    Attributes:
        positive_seeds: positive seeds of this kind of item
        negative_seeds: negative seeds of this kind of item
        landmap: landmap indicates item be stored in which map
        all_items: all items are stored here.
        used_items: all used items are stored here
        unused_items: all unused items are stored here

    """

    def __init__(self):
        self.links = {}

    def push(self, content, linktos=[]):
        if content not in self.links:
            self.links[content] = {}
        links = self.links[content]
        for linkto, count in linktos:
            links[linkto] = count


class EntityPool(ItemPool):
    """ Item pool storing items.

    Attributes:
        positive_seeds: positive seeds of this kind of item
        negative_seeds: negative seeds of this kind of item
        landmap: landmap indicates item be stored in which map
        all_items: all items are stored here.
        used_items: all used items are stored here
        unused_items: all unused items are stored here

    """

    def __init__(self):
        super(EntityPool, self).__init__()
        self.counts = {}

    def push(self, content, linktos=[]):
        if content not in self.links:
            self.links[content] = {}
            self.counts[content] = (0, 1, 0, 1)
        links = self.links[content]
        left_num, left_mean, right_num, right_mean = self.counts[content]
        left_total = left_num * left_mean
        right_total = right_num * right_mean
        for linkto, count in linktos:
            links[linkto] = count
            tag = is_legal(linkto)
            if tag == 1:
                right_num += 1
                right_total += count
            elif tag == -1:
                left_num += 1
                left_total += count
        left_mean = left_total / max(left_num, 1)
        right_mean = right_total / max(right_num, 1)
        self.counts[content] = left_num, left_mean, right_num, right_mean
