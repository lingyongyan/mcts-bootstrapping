# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-10-11 16:46:08
Last modified: 2018-12-05 09:45:50
Python release: 3.6
Notes:
"""
import torch

from core.brain import Brain


class Agent(Brain):
    def __init__(self, args, env_prototype, model_prototype,
                 memory_prototype=None):
        super(Agent, self).__init__(args, env_prototype,
                                    model_prototype, memory_prototype)
        self.hidden_dim = self.model_params.hidden_dim

        self.model_path = args.model_path
        self.model_file = args.model_file

        self.optim = args.optim
        self.device = args.device
        self.dtype = args.dtype

        self.max_steps = args.max_steps
        self.max_episode_length = args.max_episode_length
        self.gamma = args.gamma
        self.clip_grad = args.clip_grad
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.max_eval_episodes = args.max_eval_episodes
        self.prog_freq = args.prog_freq
        self.test_nepisodes = args.test_nepisodes

    def _load_model(self):
        if self.model_file:
            self.logger.warning('Loading Model:' + self.model_file + '...')
            self.model.load_state_dict(torch.load(self.model_file))
            self.logger.warning('Loaded Model:' + self.model_file + '...')
        else:
            self.logger.warning('No Pretrained Model.')

    def _save_model(self, step, curr_reward):
        self.logger.warning("Saving Model    @ Step: " +
                            str(step) + ": " + self.model_path + " ...")

        self.model.cpu()
        torch.save(self.model.state_dict(), self.model_path)
        self.model.to(self.device)
        self.logger.warning('Saved  Model    @ Step: ' +
                            str(step) + ': ' + self.model_path + '.')

    def _adjust_learning_rate(self):
        raise NotImplementedError("Not implemented in base calss")

    def _eval_model(self):  # evaluation during training
        raise NotImplementedError("Not implemented in base calss")

    def fit_model(self):    # training
        raise NotImplementedError("Not implemented in base calss")

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError("Not implemented in base calss")
