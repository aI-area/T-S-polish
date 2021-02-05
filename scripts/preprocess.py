#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/11/28 11:19:45

from multiprocessing import Pool
import pickle as pickle
import argparse
import rdkit
import os
import pandas as pd

from branch_jtnn.mol_tree import *
from util.chemutils import get_clique_mol_with_center
from branch_jtnn.mol_tree import generate_tree

# TODO: zyj
import sys
sys.setrecursionlimit(10000)
# ===================


def str_split(s, sep):
    if s.strip() == '':
        return []
    else:
        return s.split(sep)


def process(line):
    line = line.strip().split(sep=';')[:11]
    [smiles1, smiles2, scores, _, _, center1, center2, branches1, match1, matched_branch2, unmatched_branch2] = line

    scores = list(map(int, str_split(scores, ' ')))
    center1 = int(center1)
    center2 = int(center2)
    matched_branch2 = [list(set(map(int, str_split(branch, ' '))))
                       for branch in str_split(matched_branch2, ',')]
    unmatched_branch2 = [list(set(map(int, str_split(branch, ' '))))
                         for branch in str_split(unmatched_branch2, ',')]
    match1 = list(map(int, str_split(match1, ' ')))
    branches1 = [list(set(map(int, str_split(branch, ' '))))
                 for i, branch in enumerate(str_split(branches1, ','))]
    reserve_branches1 = [branch for i, branch in enumerate(branches1) if match1[i] == 1]

    reserve1 = set()
    reserve1.add(center1)
    for branch in reserve_branches1:
        reserve1.update(branch)
    reserve1 = list(reserve1)

    generate2 = set()
    generate2.add(center2)
    for branch in unmatched_branch2:
        generate2.update(branch)
    generate2 = list(generate2)

    source_mol_tree = generate_tree(smiles1, center1, assm=False)
    reserve_mol_tree = generate_tree(smiles1, center1, atoms=reserve1, assm=False)
    gen_mol_tree = generate_tree(smiles2, center2, atoms=generate2, assm=True)

    return source_mol_tree, reserve_mol_tree, gen_mol_tree, scores, center1, branches1, match1


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--mode', type=str, default='train')  # 'train'
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--split_len', type=int, default=10000)  # the surplus will be evenly allocated to each split
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.mode == 'train':
        # dataset contains molecule pairs
        with open(args.data_file) as f:
            data = f.readlines()[1:]

        with Pool(args.ncpu) as pool:
            data = pool.map(process, data)
        # data = [process(item) for item in data]

        num_splits = len(data) // args.split_len
        num_splits = 1 if num_splits == 0 else num_splits
        le = (len(data) + num_splits - 1) // num_splits
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = data[st: st + le]

            with open(os.path.join(args.save_dir, f'tensors-{split_id}.pkl'), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
