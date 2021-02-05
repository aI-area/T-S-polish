#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/11/11 09:49:24

from rdkit import Chem
import re
from multiprocessing import Pool
import argparse
import time


def get_adj_list(mol):
    adj_list = [[] for i in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_list[first].append(second)
        adj_list[second].append(first)

    return adj_list


# get all branches around the center
def get_branches(adj_list, center):
    branches = []
    has_explored = {center}

    for atom in adj_list[center]:
        if atom in has_explored:
            continue

        # dfs
        stack = [atom]
        branch = []
        while len(stack) > 0:
            cur = stack.pop()
            if cur in has_explored:
                continue

            branch.append(cur)
            has_explored.add(cur)

            for i in adj_list[cur]:
                if i not in has_explored:
                    stack.append(i)

        branches.append(branch)

    # add center atom to each branch
    for branch in branches:
        branch.append(center)

    return branches


# get canonical smiles for branch
def get_branch_smiles(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, canonical=True)
    return smiles


# compare two branches isomorphism by canonical smiles
def check_match(branch1, branch2, mol1, mol2):
    smiles1 = get_branch_smiles(mol1, branch1)
    smiles2 = get_branch_smiles(mol2, branch2)
    smiles1, _ = re.subn('\\[.(H[1-9]?)?:999\\]', '', smiles1)  # remove the center atom, to avoid the NumHs different
    smiles2, _ = re.subn('\\[.(H[1-9]?)?:999\\]', '', smiles2)
    return smiles1 == smiles2


# enumerate the branches to count score
def count_score(branches1, branches2, mol1, mol2):
    score = 0
    has_matched2 = set()
    for i in range(len(branches1)):
        for j in range(len(branches2)):
            if j in has_matched2:
                continue

            if check_match(branches1[i], branches2[j], mol1, mol2):
                has_matched2.add(j)
                score += len(branches1[i]) - 1  # -1 for the center atom
                break

    return score


# count the center scores and mapping atom for each atom in smiles1
def count_mapping_scores(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    node_num1 = mol1.GetNumAtoms()
    adj_list1 = get_adj_list(mol1)
    symbols1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    formal_charge1 = [atom.GetFormalCharge() for atom in mol1.GetAtoms()]

    mol2 = Chem.MolFromSmiles(smiles2)
    node_num2 = mol2.GetNumAtoms()
    adj_list2 = get_adj_list(mol2)
    symbols2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
    formal_charge2 = [atom.GetFormalCharge() for atom in mol2.GetAtoms()]

    final_scores = [0] * node_num1
    final_mappings = [0] * node_num1
    for i in range(node_num1):
        scores = [0] * node_num2
        for j in range(node_num2):
            if symbols1[i] != symbols2[j] or formal_charge1[i] != formal_charge2[j]:
                continue

            atom1 = mol1.GetAtomWithIdx(i)
            atom1.SetAtomMapNum(999)  # mark the center atom, force the center atom to be mapped
            atom2 = mol2.GetAtomWithIdx(j)
            atom2.SetAtomMapNum(999)

            branches1 = get_branches(adj_list1, i)
            branches2 = get_branches(adj_list2, j)
            scores[j] = count_score(branches1, branches2, mol1, mol2)

            scores[j] = scores[j] + 1  # plus one for the center atom

            atom1.SetAtomMapNum(0)
            atom2.SetAtomMapNum(0)

        final_scores[i] = max(scores)
        final_mappings[i] = scores.index(final_scores[i])

    return final_scores, final_mappings


# get the branches mapped result
def get_branches_result(smiles1, smiles2, index1, index2):
    mol1 = Chem.MolFromSmiles(smiles1)
    adj_list1 = get_adj_list(mol1)

    mol2 = Chem.MolFromSmiles(smiles2)
    adj_list2 = get_adj_list(mol2)

    atom1 = mol1.GetAtomWithIdx(index1)
    atom1.SetAtomMapNum(999)  # mark the center atom, force the center atom to be mapped
    atom2 = mol2.GetAtomWithIdx(index2)
    atom2.SetAtomMapNum(999)

    branches1 = get_branches(adj_list1, index1)
    branches2 = get_branches(adj_list2, index2)

    match1 = [0 for branch in branches1]
    has_matched2 = set()
    for i in range(len(branches1)):
        for j in range(len(branches2)):
            if j in has_matched2:
                continue

            if check_match(branches1[i], branches2[j], mol1, mol2):
                match1[i] = 1  # 1 for matched
                has_matched2.add(j)
                break

    matched_branch2 = [branches2[i] for i in has_matched2]
    unmatched_branch2 = [branches2[i] for i in range(len(branches2)) if i not in has_matched2]

    return branches1, match1, matched_branch2, unmatched_branch2


# just for the multiprocessing pool
def process_data(pair):
    smiles1, smiles2 = pair
    final_scores, final_mappings = count_mapping_scores(smiles1, smiles2)

    max_score = max(final_scores)
    max_index1 = final_scores.index(max_score)
    max_index2 = final_mappings[max_index1]

    branches1, match1, matched_branch2, unmatched_branch2 = \
        get_branches_result(smiles1, smiles2, max_index1, max_index2)

    # convert result to format string
    final_scores = ' '.join(map(str, final_scores))
    final_mappings = ' '.join(map(str, final_mappings))
    branches1 = ','.join([' '.join(map(str, branch)) for branch in branches1])
    match1 = ' '.join(map(str, match1))
    matched_branch2 = ','.join([' '.join(map(str, branch)) for branch in matched_branch2])
    unmatched_branch2 = ','.join([' '.join(map(str, branch)) for branch in unmatched_branch2])

    return '%s;%s;%s;%s;%d;%d;%d;%s;%s;%s;%s\n' % (smiles1, smiles2, final_scores, final_mappings,
                                                   max_score, max_index1, max_index2,
                                                   branches1, match1, matched_branch2, unmatched_branch2)


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_pair_file', required=True)
    parser.add_argument('--result_file', required=True)
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    # read data
    with open(args.smiles_pair_file, 'r') as f:
        pairs = f.readlines()
    pairs = [line[:-1].split(' ') for line in pairs]

    # count time
    start = time.time()

    # process data
    print('processing data...', flush=True)
    pool = Pool(args.ncpu)
    result = pool.map(process_data, pairs)
    print('process done.', flush=True)

    # save result
    with open(args.result_file, 'w') as f:
        f.write('smiles1;smiles2;scores;mappings;'
                'max_score;max_index1;max_index2;'
                'branches1;match1;matched_branch2;unmatched_branch2\n')
        f.writelines(result)

    end = time.time()
    print('time cost', end - start, 's')
