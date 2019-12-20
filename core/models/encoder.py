# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-15 14:36:34
Last modified: 2019-02-24 09:30:12
Python release: 3.6
Notes:
"""
import torch
from torchtext.data import Iterator


class Encoder(object):
    def __init__(self, args):
        self.encoder = args.encoder
        self.device = args.device
        self.batch_size = args.batch_size
        self.field = args.field
        self.dataset_cls = args.dataset_cls
        self.logger = args.logger
        self.encoder.eval()

    def encode(self, sentences, storage=None, state=None, type_='pattern',
               **kwargs):
        if not isinstance(sentences, list):
            sentences = [sentences]
        if type_ == 'entity_set':
            encoding = self._embedding(sentences)
            return encoding
            # return torch.mean(encoding, dim=0, keepdim=True)
        else:
            return self._embedding(sentences)

    def _embedding(self, sentences):
        with torch.no_grad():
            dataset = self.dataset_cls(
                sentences, fields=[('data', self.field)])
            data_iter = Iterator(dataset, batch_size=self.batch_size,
                                 sort=False, device=self.device,
                                 shuffle=False, repeat=False)
            sentences_embeddings = []
            for data in iter(data_iter):
                w_ids, lengths = getattr(data, 'data')
                weighted_embeddings = self.encoder(w_ids, lengths)
                sentences_embeddings.append(weighted_embeddings)
        return torch.cat(sentences_embeddings, dim=0)
