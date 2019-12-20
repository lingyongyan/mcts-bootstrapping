# coding=utf-8
""""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-06 10:18:06
Last modified: 2018-12-06 11:02:24
Python release: 3.6
Notes:
"""
import os
import time
import shutil

import torch
import dill
import logging


class Checkpoint(object):
    CHECKPOINT_DIR = 'checkpoints'

    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    @classmethod
    def get_last_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the
        last saved checkpoint's subdirectory.
        Precondition: at least one checkpoint has been made
        (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR)
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        if all_times:
            return os.path.join(checkpoints_path, all_times[0])
        else:
            return ''

    def dir_check(self, experiment_dir):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR,
                                  date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return path

    def save(self, experiment_dir):
        raise NotImplementedError


class OneModelCheckpoint(Checkpoint):
    def __init__(self, model, optimizer, path):
        super(OneModelCheckpoint, self).__init__(path)
        self.model = model
        self.optimizer = optimizer


class SECheckpoint(OneModelCheckpoint):
    TRAINER_STATE_NAME = 'trainer_state.pt'
    MODEL_NAME = 'model.pt'
    VOCAB_FILE = 'vocab.pt'

    def __init__(self,
                 model,
                 optimizer,
                 vocab,
                 path=None):
        super(SECheckpoint, self).__init__(model, optimizer, path)
        self.vocab = vocab
        self.logger = logging.getLogger(__name__)

    def save(self, experiment_dir):
        logger = self.logger
        path = self.dir_check(experiment_dir)
        torch.save({
            'optimizer': self.optimizer
        }, os.path.join(path, self.TRAINER_STATE_NAME))

        torch.save(self.model.state_dict(),
                   os.path.join(path, self.MODEL_NAME))

        with open(os.path.join(path, self.VOCAB_FILE), 'wb') as fout:
            dill.dump(self.vocab, fout)

        logger.info('Saved checkpoint to %s', path)
        return path

    @classmethod
    def load(cls, path, model):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object.
        """
        logger = logging.getLogger(__name__)

        resume_checkpoint = torch.load(
            os.path.join(path, cls.TRAINER_STATE_NAME),
            map_location=lambda storage, loc: storage)

        model_state = torch.load(
            os.path.join(path, cls.MODEL_NAME),
            map_location=lambda storage, loc: storage)
        model.load_state_dict(model_state)
        model.flatten_parameters()  # make RNN parameters contiguous

        with open(os.path.join(path, cls.VOCAB_FILE), 'rb') as fin:
            vocab = dill.load(fin)
        optimizer = resume_checkpoint['optimizer']

        logger.info("Loaded checkpoints from %s", path)
        return SECheckpoint(model, optimizer, vocab, path=path)
