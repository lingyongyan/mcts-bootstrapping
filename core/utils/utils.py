# coding=utf-8
""""
Program: Pytorch Utils
Description: utils for pytorch codes
Author: Lingyong Yan
Date: 2018-07-23 14:34:53
Last modified: 2019-12-20 09:20:54
Python release: 3.6
"""
import logging
import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


def poisson(lmbd):
    L, k, p = math.exp(-lmbd), 0, 1
    while p > L:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)


###############################################################################
# pytorch network utils
###############################################################################
def net_weight_init(model):
    """
    init the weight of pytorch networks

    :param model:

    """
    if isinstance(model, nn.Conv2d):
        n = model.kernel_size[0] * model.kernel_size[1] * model.out_channels
        nn.init.xavier_normal_(model.weight, gain=math.sqrt(2. / n))
        logging.debug('initialze conv layer')
    elif isinstance(model, nn.Linear):
        size = model.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = math.sqrt(2. / (fan_out + fan_in))
        nn.init.xavier_normal_(model.weight, gain=variance)
        logging.debug('initialze linear layer')
    else:
        logging.debug('other module, name:%s', model.__class__.__name__)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_shared_network(args, T, model,
                          shared_model, shared_average_model,
                          loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
    transfer_grad_to_shared_model(model, shared_model)
    optimizer.step()
    if args.lr_decay:
        adjust_learning_rate(optimizer, max(
            args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))
    for shared_param, shared_average_param in zip(shared_model,
                                                  shared_average_model):
        shared_average_param = args.trust_region_decay * \
            shared_average_param + (1 - args.trust_region_decay) * shared_param

###############################################################################
# reinforcement learning utils
###############################################################################


def generalized_advantage_estimation(rewards, values, gamma, tau, device):
    gae_ts = torch.tensor(np.zeros_like(rewards),
                          device=device,
                          dtype=torch.float)
    gae_t = torch.zeros((1, 1), device=device)
    for t in reversed(range(len(rewards))):
        tderr_ts = rewards[t] + gamma * values[t+1].data - values[t].data
        gae_t = gamma * tau * gae_t + tderr_ts
        gae_ts[t] = gae_t
    return gae_ts


def categorical_kl_div(distribution, ref_distribution):
    kl = (ref_distribution * (ref_distribution.log() -
                              distribution.log())).sum(1).mean(0)
    return kl


def trust_region_loss(policy_loss, policy, avg_policy, threshold):
    kl_div = categorical_kl_div(policy, avg_policy)
    k = grad(outputs=kl_div, inputs=policy,
             retain_graph=False, only_inputs=True)[0]
    g = grad(outputs=policy_loss, inputs=policy,
             retain_graph=False, only_inputs=True)[0]
    k_dot_g = (k * g).sum(1).mean(0)
    k_dot_k = (k * k).sum(1).mean(0)
    if k_dot_k.item() > 0:
        trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0).detach()
    else:
        trust_factor = torch.zeros(1)
    trust_region_grad = g - trust_factor * k
    return trust_region_grad


def TD_rollout(env, agent, init_state, steps=1):
    actions = []
    log_probs = []
    values = []
    rewards = []
    entropies = []
    final_r = 0.0
    state = init_state
    is_done = False

    logging.info('TD sampling for [%d] steps' % (steps))
    for step in range(steps):
        action, log_prob, value, action_probs = agent.select_action(state)
        entropy = - (action_probs * action_probs.log()).sum()
        probs = action_probs.detach().view(-1).cpu().numpy().tolist()
        data = [(p.content, prob) for p, prob in zip(env.next_patterns, probs)]
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
        print('----------top 10----------')
        for pattern, prob in sorted_data[:10]:
            print('%s: %.3f' % (pattern, prob))
        print('----------bottom 10----------')
        for pattern, prob in sorted_data[-10:]:
            print('%s: %.3f' % (pattern, prob))
        next_state, reward, done = env.step(action)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)
        state = next_state
        if done:
            is_done = True
            agent.model.eval()
            _, final_r = agent.model(next_state)
            agent.model.train()
            break
    return actions, log_probs, values, rewards, entropies, final_r,\
        state, is_done


def pretrain(env, model, init_state, path):
    states = [[] for i in range(3)]
    actions = []
    state = init_state

    logging.info('Do pretrain')
    for i, pattern in enumerate(path):
        states[0].append(state[0])
        states[1].append(state[1])
        states[2].append(state[2])
        pattern_contents = env.get_pattern_contents()
        if pattern not in pattern_contents:
            env.print_patterns()
            raise ValueError('pattern "%s" must in pattern list' % pattern)
        action = pattern_contents.index(pattern)
        next_state = env.step(action)
        actions.append(action)
        state = next_state
    return states, actions, state


def pad_sequence(features, max_len, PAD=0):
    padded_features = []
    lengths = []
    for feature in features:
        if len(feature) >= max_len:
            padded_feature = feature[:max_len]
            length = max_len
        else:
            length = len(feature)
            padded_feature = np.pad(
                feature, ((0, max_len - length), (0, 0)), 'constant')
        padded_features.append(padded_feature)
        lengths.append(length)
    return padded_features, lengths


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def transfer_grad_to_shared_model(model, shared_model):
    for param, shared_param in zip(model.paramters(), shared_model.parameters):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
