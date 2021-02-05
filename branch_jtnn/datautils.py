#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/11/29 10:08:11

import os
import pickle as pickle
import random

import torch
from torch.utils.data import Dataset, DataLoader

import logging

from branch_jtnn.jtmpn import JTMPN
from branch_jtnn.jtnn_enc import JTNNEncoder
from branch_jtnn.mpn import MPN
from branch_jtnn.branch_jtnn import tensorize_tree


class TrainDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1, batch2, scores, centers, branches, match = list(zip(*self.data[idx]))
        return tensorize_tree(batch0, self.vocab, assm=False), tensorize_tree(batch1, self.vocab, assm=False), \
               tensorize_tree(batch2, self.vocab, assm=self.y_assm), scores, centers, branches, match


class TrainFolder(object):

    def __init__(self, data_dir, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True):
        self.data_dir = data_dir
        self.data_files = [fn for fn in os.listdir(data_dir) if fn.endswith('.pkl') and fn.startswith('tensors-')]
        self.data_files = sorted(self.data_files, key=lambda s: int(s[8:-4]))
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle
        self.logger = logging.getLogger('logger')

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_dir, fn)
            self.logger.info(f'Open data file: {fn}')
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = TrainDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                    collate_fn=lambda x: x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader
