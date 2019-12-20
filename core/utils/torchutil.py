# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-05 16:18:25
Last modified: 2019-08-25 16:13:27
Python release: 3.6
Notes:
"""
import numpy as np
import torch


def calc_reward(stats, gamma, following_stats=None):
    assert len(stats) == 4 # new_sum, new_count, old_sum, old_count
    if following_stats:
        assert len(following_stats) == 3 # reward, new_sum, new_count
        weight_sum = following_stats[1] * gamma
        weight_count = following_stats[2] * gamma
    else:
        weight_sum = 0.
        weight_count = 0.
    weight_sum += stats[0]
    weight_count += stats[1]
    if weight_count > 0.:
        reward = weight_sum / weight_count * reward_weight(weight_count + 1)
    else:
        reward = 0
    return reward, weight_sum, weight_count


def discounted_reward(stats_list, gamma, final_return=None):
    discounted = [0. for _ in range(len(stats_list))]
    following_stats = final_return
    for t in reversed(range(0, len(stats_list))):
        following_stats = calc_reward(stats_list[t], gamma, following_stats)
        discounted[t] = following_stats
    return discounted


def reward_weight(count):
    return sigmoid(count / 2)


def sigmoid(x):
    s = 1 / (1+np.exp(-x))
    return s


def weight_adaption(orig_values):
    return orig_values


def weight_normalize(orig_weights, min_v=None, max_v=None):
    weights = np.array(orig_weights, dtype=np.float)
    all_weight = weights.sum()
    if all_weight > 0.:
        weights /= all_weight
        if min_v is not None or max_v is not None:
            weights = np.clip(weights, min_v, max_v)
            weights /= weights.sum()
    else:
        weights = np.ones(weights.shape[0]) / weights.shape[0]
    return weights


def weight_normalize_with_clip(orig_weights):
    weights = weight_normalize(orig_weights)
    return weights


def sequence_mask(lens, max_len=None):
    """get a mask matrix from batch lens variables

    :param lens:
    :param max_len:  (Default value = None)

    """
    if max_len is None:
        max_len = lens.max().item()
    batch_size = lens.size(0)

    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_broadcast = lens.unsqueeze(-1).expand_as(ranges)
    mask = ranges < lens_broadcast
    return mask


def mask_mean_weights(mask):
    new_mask = mask.float()
    sum_mask = new_mask.sum(dim=1, keepdim=True)
    indice = (sum_mask > 0).squeeze(1)
    new_mask[indice] /= sum_mask[indice]
    return new_mask


if __name__ == '__main__':
    lengths = [1, 2, 3, 4, 5]
    length_vec = torch.tensor(lengths, dtype=torch.long)
    mask = sequence_mask(length_vec)
    print(mask)
    new_mask = mask_mean_weights(mask)
    print(new_mask)
