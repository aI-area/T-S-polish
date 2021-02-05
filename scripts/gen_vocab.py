#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/11/29 11:20:37

import rdkit
import rdkit.Chem as Chem
import copy
import sys
import argparse
from multiprocessing import Pool

from util.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from branch_jtnn.mol_tree import MolTree

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--mol_file', required=True)
parser.add_argument('--save_vocab_file', required=True)
parser.add_argument('--ncpu', type=int, default=8)
args = parser.parse_args()

cset = set()
with open(args.mol_file, 'r') as f:
    lines = f.readlines()
lines = [line[:-1] for line in lines]


def get_tree(smiles):
    return MolTree(smiles, 0)

pool = Pool(args.ncpu)
trees = pool.map(get_tree, lines)

for smiles in lines:
    mol = Chem.MolFromSmiles(smiles)
    for i in range(mol.GetNumAtoms()):
        cmol = get_clique_mol(mol, [i])
        cset.add(get_smiles(cmol))
for tree in trees:
    for c in tree.nodes:
        cset.add(c.smiles)

result = '\n'.join(sorted(cset))
with open(args.save_vocab_file, 'w') as f:
    f.write(result)

print('done', flush=True)
