# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-08-29 10:51:59
Last modified: 2019-12-20 09:14:11
Python release: 3.6
"""
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from .settings import device, eps
from .utils import discount_reward, trust_region_loss, update_shared_network
from .utils import poisson
from .modules import EpisodicReplayMemory


class Learner(object):
    def __init__(self, model, gamma, device, seed=None):
        self.model = model
        self.gamma = gamma
        self.device = device
        self.model.to(device)
        if seed:
            self.seed(seed)

    def to(self, device):
        self.model.to(device)

    def save(self, path='./save/model.pkl'):
        torch.save(self.model.state_dict(), path)

    def load(self, path='./save/model.pkl'):
        self.model.load_state_dict(torch.load(path))

    def seed(self, seed):
        torch.manual_seed(seed)


class A2C(Learner):
    def __init__(self, model, gamma, device, seed=None):
        super(A2C, self).__init__(model, gamma, device, seed=seed)
        self.critic_criterion = nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger = logging.getLogger(__name__)
        self.model.train()

    def select_action(self, state):
        action_probs, state_value = self.model(state)
        action = action_probs.multinomial(1).view(-1, 1)
        action_logs = action_probs.log()
        log_prob = torch.gather(action_logs, dim=-1, index=action)
        return action.item(), log_prob, state_value, action_probs

    def update(self, log_probs, values, rewards, entropies, final_v):
        discount_rewards = discount_reward(rewards, self.gamma, final_v)
        discount_rewards = torch.Tensor(discount_rewards).to(device)
        discount_rewards = (
            discount_rewards - discount_rewards.mean())\
            / (discount_rewards.std() + eps)
        log_probs = torch.stack(log_probs).view(-1, 1)
        entropies = torch.stack(entropies).view(-1, 1)
        values = torch.stack(values).view(-1, 1)
        advantages = discount_rewards - values
        policy_loss = -torch.mean(log_probs * advantages + 1e-3 * entropies)
        value_loss = torch.mean(
            self.critic_criterion(values, discount_rewards))
        loss = policy_loss + value_loss
        self.optim.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()


class REINFORCE(Learner):
    def __init__(self, model, gamma, device):
        super(REINFORCE, self).__init__(model, gamma, device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        probs = self.model(state)
        action = probs.multinomial(1)
        prob = probs[:, action[0, 0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs * probs.log()).sum()
        return action[0].item(), log_prob, entropy

    def update(self, log_probs, entropies, rewards):
        R = torch.zeros(1, 1)
        loss = 0
        for reward, log_prob, entropy in\
                reversed(list(zip(rewards, log_probs, entropies))):
            R = self.gamma * R + reward
            tensor_R = torch.Tensor(R).to(device)
            loss = loss - (log_prob * tensor_R.expand_as(log_prob)
                           ).sum() - (0.0001 * entropy).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()


def train(rank, args, T, Environment, Agent,
          shared_model, shared_average_model,
          optimizer):
    # , externals, ps, ns, pattern_filter):
    model = Agent(args.pattern_size,
                  args.feature_size,
                  args.pattern_size)
    # model.to(device)
    model.train()
    # env = Environment(externals, ps, ns, pattern_filter, args.pattern_size,
    #                  args.max_episode_length, None)
    env = Environment()

    if not args.on_policy:
        memory = EpisodicReplayMemory(
            args.memory_capacity // args.num_processes,
            args.max_episode_length)

    t = 1
    done = True

    while T.value() <= args.T_max:
        while True:
            model.load_state_dict(shared_model.state_dict())
            t_start = t

            if done:
                state = env.reset()
                done, episode_length = False, 0
            policies, Qs, Vs = [], [], []
            actions, rewards, average_policies = [], [], []

            while not done and t - t_start < args.t_max:
                policy, Q, V = model(state)
                average_policy, _, _ = shared_average_model(state)
                action = policy.multinomial(1)
                next_state, reward, done = env.step(action.item())
                done = done or episode_length >= args.max_episode_length
                episode_length += 1

                if not args.on_policy:
                    memory.append(state, action, reward, policy.detach())
                [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions,
                                                    rewards, average_policies),
                                                   (policy, Q, V, action,
                                                    reward, average_policy))]
                t += 1
                T.increment()
                state = next_state
            if done:
                Qret = torch.zeros(1, 1)
                if not args.on_policy:
                    memory.append(state, None, None, None)
            else:
                _, _, Qret = model(state)
                Qret = Qret.detach()
            _train(args, T, model, shared_model, shared_average_model,
                   optimizer, policies, Qs, Vs, actions, rewards,
                   Qret, average_policies)

            if done:
                break

        if not args.on_policy and len(memory) >= args.replay_start:
            for _ in range(poisson(args.replay_ratio)):
                trajectories = memory.sample_batch(
                    args.batch_size, maxlen=args.t_max)

                policies, Qs, Vs, actions, rewards = [], [], [], [], []
                old_policies, average_policies = [], []

                for i in range(len(trajectories - 1)):
                    state = torch.cat(tuple(
                        trajectory.state for trajectory in trajectories[i]), 0)
                    action = torch.LongTensor([trajectory.action for
                                               trajectory in trajectories[i]
                                               ]).unsqueeze(1)
                    reward = torch.Tensor([trajectory.reward for
                                           trajectory in trajectories[i]
                                           ]).unsqueeze(1)
                    old_policy = torch.cat(tuple(trajectory.policy for
                                                 trajectory in trajectories[i]
                                                 ), 0)

                    policy, Q, V = model(state)
                    average_policy, _, _ = shared_average_model(state)

                    [arr.append(el) for arr, el in zip((policies, Qs, Vs,
                                                        actions, rewards,
                                                        average_policies,
                                                        old_policies),
                                                       (policy, Q, V, action,
                                                        reward, average_policy,
                                                        old_policy))]
                    next_state = torch.cat(tuple(trajectory.state for
                                                 trajectory in
                                                 trajectories[i + 1]), 0)
                    done = torch.Tensor([trajectory.action is None
                                         for trajectory in trajectories[i + 1]
                                         ]).unsqueeze(1)

                _, _, Qret = model(next_state)
                Qret = ((1 - done) * Qret).detach()
                _train(args, T, model, shared_model, shared_average_model,
                       optimizer, policies, Qs, Vs, actions, rewards,
                       Qret, average_policies, old_policies=old_policies)
        done = True
    # env.close()


def _train(args, T, model, shared_model, shared_average_model, optimizer,
           policies, Qs, Vs, actions, rewards, Qret,
           average_policies, old_policies=None):
    off_policy = old_policies is not None
    action_size = policies[0].size()
    policy_loss, value_loss = 0, 0
    t = len(rewards)
    for i in reversed(range(t)):
        if off_policy:
            rho = policies[i].detach() / old_policies[i]
        else:
            rho = torch.ones(1, action_size)
        Qret = rewards[i] + args.gamma * Qret
        A = Qret - Vs[i]

        log_prob = policies[i].gather(1, actions[i]).log()
        single_step_policy_loss = - \
            (rho.gather(1, actions[i]).clamp(
                max=args.trace_max) * log_prob * A.detach()).mean(0)
        if off_policy:
            bias_weight = (1 - args.trace_max / rho).clamp(min=0) * policies[i]
            single_step_policy_loss -= (
                bias_weight * policy_loss[i].log() *
                (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())
            ).sum(1).mean(0)
        if args.trust_region:
            k = - average_policies[i].gather(1, actions[i]) / \
                (policies[i].gather(1, actions[i]) + 1e-10)
            if off_policy:
                g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A /
                     (policies[i] + 1e-10).gather(1, actions[i])
                     +
                     (bias_weight * (Qs[i] - Vs[i].expand_as(Qs[i])) /
                      (policies[i] + 1e-10)).sum(1)).detach()
            else:
                g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A /
                     (policies[i] + 1e-10).gather(1, actions[i])).detach()
            policy_loss += trust_region_loss(
                model,
                policies[i].gather(1, actions[i]) + 1e-10,
                average_policies[i].gather(1, actions[i]) + 1e-10,
                single_step_policy_loss, args.trust_region_threshold, g, k)
        else:
            policy_loss += single_step_policy_loss
        policy_loss += args.entropy_weight * \
            (policies[i].log() * policies[i]).sum(1).mean(0)

        Q = Qs[i].gather(1, actions[i])
        value_loss += ((Qret - Q) ** 2 / 2).mean(0)
        truncated_rho = rho.gather(1, actions[i]).clamp(max=1)
        Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()
        update_shared_network(args, T, model, shared_model,
                              shared_average_model, policy_loss + value_loss,
                              optimizer)
