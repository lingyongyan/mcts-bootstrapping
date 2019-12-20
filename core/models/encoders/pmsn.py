# coding=utf-8
"""
Program:
Description:
Author: Lingyong Yan
Date: 2018-12-19 20:40:44
Last modified: 2019-08-25 04:32:56
Python release: 3.6
Notes:
"""
import torch
from core.models.encoder import Encoder
from core.models.encoders.dpe import DPE


class PMSN(Encoder):
    def __init__(self, args):
        super(PMSN, self).__init__(args)
        self.context_count = args.context_count
        self.dpe = DPE()

    def encode(self, patterns, storage=None, state=None, type_='pattern'):
        if type_ == 'pattern':
            batch = len(patterns)
            embeddings = self._embedding(patterns).unsqueeze(1)
            weights = torch.ones((batch, 1), device=self.device,
                                 dtype=torch.float)
            return embeddings, weights
        elif type_ == 'entity' or type_ == 'entity_set':
            p_ws = self.dpe.dsm_init_weight(storage, state, patterns,
                                            self.context_count, type_=type_)
            count = len(p_ws)
            ps = []
            ws = []
            for p, w in p_ws:
                ps.extend(p)
                ws.extend(w)
            embs = self._embedding(ps).view(count, self.context_count, -1)
            ws = torch.tensor(ws, device=self.device,
                              dtype=torch.float).view(count, self.context_count)
            return embs, ws
        else:
            raise ValueError('type should be in ("entity", "pattern")')
