# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-17 16:25:53
Last modified: 2019-09-01 21:44:20
Python release: 3.6
Notes:
"""
import math
import heapq

from core.utils.torchutil import weight_normalize


def get_dpe_weight(n_neighbors, n_matched):
    weight = n_matched * math.log(n_matched + 1) / n_neighbors
    return weight


def rlogf(n_neighbors, pos_matched, neg_matched):
    weight = (pos_matched - neg_matched) * math.log(pos_matched + 1) / n_neighbors
    return weight


def get_expanding_patterns(storage, entities):
    patterns = set()
    for e in entities:
        for p, count in storage.entity_pool.links[e].items():
            patterns.add(p)
    return list(patterns)


def sort_expanding_patterns_by_rlogf(state, storage, num=100):
    """
    use rlogf function to filter top_num patterns (with weights)

    :param storage:
    :param state:
    :param num:

    """
    patterns = []
    for p, (n_neighbors, pos_matched, neg_matched) in state.extracted_patterns.items():
        if pos_matched < 1:
            continue
        weight = rlogf(n_neighbors, pos_matched, neg_matched)
        if weight > 0:
            patterns.append((p, weight))

    items = heapq.nlargest(num, patterns, key=lambda x: x[1])

    patterns = [p[0] for p in items]
    pattern_weights = [p[1] for p in items]
    # pattern_weights = weight_normalize(pattern_weights)
    return patterns, pattern_weights


def get_adpe(storage, entities, context_count=100, adaptions=None):
    patterns = _get_adpe(storage, entities, context_count=context_count,
                         adaptions=adaptions)
    patterns = _post_process(patterns, context_count)
    return patterns


def _get_adpe(storage, entities, context_count=100, adaptions=None):
    """
    use counting stastics to filter core patterns of entities (without weights)

    :param storage:
    :param entities:
    :param num:  (Default value = 500)

    """
    patterns = []
    n_entities = len(entities)
    link_patterns = storage.get_normal_patterns_by_entities(entities, 0.1)

    for p, counts in link_patterns.items():
        match = len(counts)
        link_es = storage.pattern_pool.links[p]
        df = len(link_es)
        if n_entities > 1 and match < 2:
            continue
        else:
            weight = get_dpe_weight(df, sum(counts) / n_entities)
            if adaptions and p in adaptions:
                weight *= adaptions[p]
        patterns.append((p, weight, df, match))

    if context_count > 0:
        patterns = heapq.nlargest(context_count, patterns, key=lambda x: x[1])
    return patterns


def _post_process(patterns, num=None):
    weights = [p[1] if p[1] > 0. else 0. for p in patterns]
    patterns = [p[0] for p in patterns]
    if len(patterns) == 0:
        patterns = ['<unk>']
        weights = [0.]
    if num is not None:
        for i in range(num - len(patterns)):
            patterns.append('<unk>')
            weights.append(0.)
    return patterns, weights

# home_path = os.getcwd()

# DOC_NUM = 41242
# SEN_PATTERN = re.compile(r'[a-zA-Z \']+$')


'''
def get_set_idf(storage, patterns, beaf=0):
    entities = set()
    if not isinstance(patterns, list):
        patterns = [patterns]
    for p in patterns:
        entities.update(storage.pattern_pool.links[p].keys())
    idf = math.log((DOC_NUM - beaf) / max(len(entities) - beaf, 1))
    return idf


def sort_semantic_patterns_set(storage, entities,
                               context_count=500, adaptions=None):
    results = sort_patterns_by_count_set(
        storage,
        entities,
        context_count=context_count,
        adaptions=adaptions)
    return results


def sort_patterns_by_count_set(storage, entities, num=200, adaptions=None):
    """
    use counting stastics to filter core patterns of entities (without weights)

    :param storage:
    :param entities:
    :param num:  (Default value = 500)

    """
    results = []
    patterns = _inner_sort_by_count_set(storage, entities, adaptions=adaptions)
    for e in entities:
        left_items = []
        right_items = []
        for p in storage.entity_pool.links[e].keys():
            if p in patterns:
                if p[0] == '_':
                    right_items.append((p, patterns[p]))
                else:
                    left_items.append((p, patterns[p]))
        if num > 0:
            left_items = heapq.nlargest(num, left_items, key=lambda x: x[1])
            right_items = heapq.nlargest(num, right_items, key=lambda x: x[1])

        left_patterns, left_weights = _post_process(left_items, num)
        right_patterns, right_weights = _post_process(right_items, num)
        right_weights = weight_normalize(right_weights)
        left_weights = weight_normalize(left_weights)
        results.append((left_patterns, left_weights,
                        right_patterns, right_weights))
    return results

def _inner_sort_by_count_set(storage, entities, adaptions=None):
    all_num = len(entities)
    link_patterns = storage.get_normal_patterns_by_entities(entities, 0.05)
    patterns = {}
    for p, counts in link_patterns.items():
        match = len(counts)
        link_es = storage.pattern_pool.links[p]
        df = len(link_es)
        if match < 2 or df < 10 or df > 1200:
            continue
        else:
            weight = get_dpe_weight(sum(counts) / all_num, df)
            if adaptions and p in adaptions:
                weight *= adaptions[p]
            patterns[p] = weight
    return patterns
'''


def sort_expanding_patterns_by_meb(policy, state, storage, num=100):
    """
    use meb function to filter top_num patterns (with weights)

    :param storage:
    :param state:
    :param num:  (Default value = 100)

    """
    # all_pos = len(state.top) + len(state.pos_seeds)
    patterns = []
    for p, (df, pos_matched, neg_matched) in state.extracted_patterns.items():
        weight = rlogf(df, pos_matched, neg_matched)
        patterns.append((p, weight))

    items = heapq.nlargest(num * 2, patterns, key=lambda x: x[1])

    ps = [p for p, _ in items]
    probs, _ = policy.predict(state, ps)

    pattern_probs = [(p, w) for p, w in zip(ps, probs)]

    items = heapq.nlargest(num, pattern_probs, key=lambda x: x[1])

    patterns = [p[0] for p in pattern_probs]
    pattern_weights = [p[1] for p in pattern_probs]
    pattern_weights = weight_normalize(pattern_weights)
    return patterns, pattern_weights


def sort_by_direct_count(items, item_pool, other_pool,
                         filter_func=None, num=500):
    """
    use counting stastics to filter core patterns of entities (without weights)

    :param storage:
    :param entities:
    :param num:  (Default value = 500)

    """
    target_items = {}
    total_n = len(items)
    for i in items:
        for other_i, count in item_pool.links[i].items():
            if filter_func is None or filter_func(other_i) is not None:
                if other_i not in target_items:
                    target_items[other_i] = (1, count)
                else:
                    n, old_count = target_items[other_i]
                    target_items[other_i] = (n + 1, count + old_count)
    for i in target_items:
        p_n = len(other_pool.links.get(i, []))
        n, count = target_items[i]
        weights = n / (total_n * math.log2(p_n + 1)) if p_n > 0 else 0.
        target_items[i] = (n, count, weights)
    target_items = sorted(target_items.items(),
                          key=lambda x: (x[1][0], x[1][1]), reverse=True)
    target_items = target_items[:num]
    return target_items


def sort_core_patterns_by_count(storage, entities, num=200):
    """
    use counting stastics to filter core patterns of entities (without weights)

    :param storage:
    :param entities:
    :param num:  (Default value = 500)

    """
    left_patterns = []
    right_patterns = []
    all_num = len(entities)
    half = math.ceil(all_num / 2)
    link_patterns = storage.get_normal_patterns_by_entities(entities, 0.1)

    for p, counts in link_patterns.items():
        match = len(counts)
        link_es = storage.pattern_pool.links[p]
        df = len(link_es)
        # beaf = df - match
        if (all_num > 1 and match < half) or df < 10 or df > 1000:
            continue
        else:
            weight = get_dpe_weight(sum(counts) / all_num, df)
        if p[0] == '_':
            right_patterns.append((p, match, df, weight))
        else:
            left_patterns.append((p, match, df, weight))
    if num > 0:
        left_patterns = heapq.nlargest(num, left_patterns, key=lambda x: x[-1])
        right_patterns = heapq.nlargest(
            num, right_patterns, key=lambda x: x[-1])
    return left_patterns + right_patterns


###############################################################################
# Filter functions
###############################################################################
'''
def semantic_pattern_filter(pattern):
    # we filter the pattern is particular
    # we filter out general patterns.
    splits = pattern.strip().split()
    if splits[0] == '_':
        if not splits[1][0].isupper():
            return is_legal(splits[1:])
    else:
        if not splits[3][0].isupper():
            return is_legal(splits[:4])
    return 0


ALL_STOP_WORDS_FILE = os.path.join(home_path, 'data/all_stopwords.txt')
all_stop_words = set(nltk.corpus.stopwords.words('english'))
all_stop_words.update(load_data(ALL_STOP_WORDS_FILE))

LEGAL_PART = ('IN', 'MD', 'RP', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')


def is_legal(pattern_splits):
    for w in pattern_splits:
        if w.lower() not in all_stop_words:
            return 1
    return -1
'''

###############################################################################
# Weight precess functions
###############################################################################


###############################################################################
# sort functions
###############################################################################
