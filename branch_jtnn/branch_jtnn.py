#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/11/29 14:19:59

import torch
import torch.nn as nn
import torch.nn.functional as F
from branch_jtnn.mol_tree import Vocab, MolTree
from util.nnutils import create_var, flatten_tensor, avg_pool
from branch_jtnn.jtnn_enc import JTNNEncoder
from branch_jtnn.jtnn_dec import JTNNDecoder
from branch_jtnn.mpn import MPN
from branch_jtnn.jtmpn import JTMPN
from branch_jtnn.center_pred import CenterPredictor
from branch_jtnn.branch_pred import BranchPredictor
from util.nnutils import *

from util.chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, copy_atom
import rdkit
import rdkit.Chem as Chem
import copy
import math
import logging
from util.chemutils import get_mol
from util.chemutils import get_branches
from branch_jtnn.mol_tree import generate_tree


class BranchJTNN(nn.Module):

    def __init__(self, vocab, args):
        super(BranchJTNN, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size = args.hidden_size
        self.rand_size = rand_size = args.rand_size

        mpn_dropout_rate = args.mpn_dropout_rate if 'mpn_dropout_rate' in args else 0.0
        self.mpn = MPN(hidden_size, args.depthG, mpn_dropout_rate)

        # ================= predict center =========================
        self.center_predictor = CenterPredictor(hidden_size)
        # ==========================================================

        # ================= predict branch =========================
        self.branch_predictor = BranchPredictor(hidden_size)
        # ==========================================================

        # ================= generate branch ========================
        if args.share_embedding:
            self.embedding = nn.Embedding(vocab.size(), hidden_size)
            self.jtnn = JTNNEncoder(hidden_size, args.depthT, self.embedding)
            self.decoder = JTNNDecoder(vocab, hidden_size, self.embedding, args.use_molatt)
        else:
            self.jtnn = JTNNEncoder(hidden_size, args.depthT, nn.Embedding(vocab.size(), hidden_size))
            self.decoder = JTNNDecoder(vocab, hidden_size, nn.Embedding(vocab.size(), hidden_size), args.use_molatt)

        self.jtmpn = JTMPN(hidden_size, args.depthG)

        self.A_assm = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(reduction='mean')

        self.T_mean = nn.Linear(hidden_size, rand_size // 2)
        self.T_var = nn.Linear(hidden_size, rand_size // 2)
        self.G_mean = nn.Linear(hidden_size, rand_size // 2)
        self.G_var = nn.Linear(hidden_size, rand_size // 2)
        self.B_t = nn.Sequential(nn.Linear(hidden_size + rand_size // 2, hidden_size), nn.ReLU())
        self.B_g = nn.Sequential(nn.Linear(hidden_size + rand_size // 2, hidden_size), nn.ReLU())

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(
            *jtenc_holder)  # 树的特征(batch_size * 树最大节点数 * hidden_size) message特征(边数量 * hidden_size)
        mol_vecs = self.mpn(*mpn_holder)  # 图的特征(batch_size * 图最大节点数 * hidden_size)
        return tree_vecs, tree_mess, mol_vecs

    # def fuse_noise(self, tree_vecs, mol_vecs):
    #     tree_eps = create_var( torch.randn(tree_vecs.size(0), 1, self.rand_size / 2) )
    #     tree_eps = tree_eps.expand(-1, tree_vecs.size(1), -1)
    #     mol_eps = create_var( torch.randn(mol_vecs.size(0), 1, self.rand_size / 2) )
    #     mol_eps = mol_eps.expand(-1, mol_vecs.size(1), -1)
    #
    #     tree_vecs = torch.cat([tree_vecs,tree_eps], dim=-1)
    #     mol_vecs = torch.cat([mol_vecs,mol_eps], dim=-1)
    #     return self.B_t(tree_vecs), self.B_g(mol_vecs)

    def fuse_tree_noise(self, tree_vecs):
        tree_eps = create_var(torch.randn(tree_vecs.size(0), 1, self.rand_size // 2))
        tree_eps = tree_eps.expand(-1, tree_vecs.size(1), -1)
        cat_tree_vecs = torch.cat([tree_vecs, tree_eps], dim=-1)

        return self.B_t(cat_tree_vecs)

    def fuse_mol_noise(self, mol_vecs):
        mol_eps = create_var(torch.randn(mol_vecs.size(0), 1, self.rand_size // 2))
        mol_eps = mol_eps.expand(-1, mol_vecs.size(1), -1)
        cat_mol_vecs = torch.cat([mol_vecs, mol_eps], dim=-1)

        return self.B_g(cat_mol_vecs)

    def fuse_pair(self, x_tree_vecs, x_mol_vecs, y_tree_vecs, y_mol_vecs, jtenc_scope, mpn_scope):
        diff_tree_vecs = y_tree_vecs.sum(dim=1) - x_tree_vecs.sum(dim=1)
        size = create_var(torch.Tensor([le for _,le in jtenc_scope]))
        diff_tree_vecs = diff_tree_vecs / size.unsqueeze(-1)

        diff_mol_vecs = y_mol_vecs.sum(dim=1) - x_mol_vecs.sum(dim=1)
        size = create_var(torch.Tensor([le for _,le in mpn_scope]))
        diff_mol_vecs = diff_mol_vecs / size.unsqueeze(-1)

        diff_tree_vecs, tree_kl = self.rsample(diff_tree_vecs, self.T_mean, self.T_var)
        diff_mol_vecs, mol_kl = self.rsample(diff_mol_vecs, self.G_mean, self.G_var)

        diff_tree_vecs = diff_tree_vecs.unsqueeze(1).expand(-1, x_tree_vecs.size(1), -1)
        diff_mol_vecs = diff_mol_vecs.unsqueeze(1).expand(-1, x_mol_vecs.size(1), -1)
        cat_x_tree_vecs = torch.cat([x_tree_vecs, diff_tree_vecs], dim=-1)
        cat_x_mol_vecs = torch.cat([x_mol_vecs, diff_mol_vecs], dim=-1)

        # return self.B_t(cat_x_tree_vecs), self.B_g(cat_x_mol_vecs), tree_kl + mol_kl
        return self.B_t(cat_x_tree_vecs), self.B_g(cat_x_mol_vecs), tree_kl + mol_kl

    def rsample(self, z_vecs, W_mean, W_var):
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.mean(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, x_batch, reserve_x_batch, y_batch, scores, centers, branches, matches, beta):
        x_batch, x_jtenc_holder, x_mpn_holder = x_batch
        reserve_x_batch, reserve_x_jtenc_holder, reserve_x_mpn_holder = reserve_x_batch
        y_batch, y_jtenc_holder, y_mpn_holder, y_jtmpn_holder = y_batch

        # encoding
        x_tree_vecs, _, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        reserve_x_tree_vecs, _, reserve_x_mol_vecs = self.encode(reserve_x_jtenc_holder, reserve_x_mpn_holder)
        y_tree_vecs, y_tree_mess, y_mol_vecs = self.encode(y_jtenc_holder, y_mpn_holder)

        # fuse pair
        x_tree_vecs, x_mol_vecs, kl_div = self.fuse_pair(x_tree_vecs, x_mol_vecs, y_tree_vecs, y_mol_vecs,
                                                         y_jtenc_holder[-1], y_mpn_holder[-1])

        # predict center
        mpn_scope = x_mpn_holder[-1]
        atom_num = [item[1] for item in mpn_scope]
        center_loss, center_correct_num, center_total_num = self.center_predictor(x_mol_vecs, atom_num, scores)

        # predict branch
        branch_loss, branch_tp, branch_fp, branch_fn, branch_tn = self.branch_predictor(x_mol_vecs, centers, branches, matches)

        # generate branch
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(y_batch, x_tree_vecs, x_mol_vecs,
                                                                reserve_x_tree_vecs, reserve_x_mol_vecs)
        assm_loss, assm_acc = self.assm(y_batch, y_jtmpn_holder, x_mol_vecs, reserve_x_mol_vecs, y_tree_mess)

        return center_loss + branch_loss + word_loss + topo_loss + assm_loss + beta * kl_div, \
               kl_div.item(), center_correct_num, center_total_num, branch_tp, branch_fp, branch_fn, branch_tn, \
               word_acc, topo_acc, assm_acc

    def pipeline_predict(self, smiles):
        # predict center
        mol = get_mol(smiles)
        atom_num = mol.GetNumAtoms()
        atoms = list(range(atom_num))
        mpn_holder = MPN.tensorize([smiles], [atoms])
        x_mol_vecs = self.mpn(*mpn_holder)
        # fuse mol noise
        x_mol_vecs = self.fuse_mol_noise(x_mol_vecs)
        scores = self.center_predictor.get_predict_score(x_mol_vecs, [atom_num])
        center = scores[0].argmax().item()

        # predict branch
        branches = get_branches(mol, center)
        predict = self.branch_predictor.predict(x_mol_vecs, [center], [branches])
        predict = [int(item >= 0.5) for item in predict[0]]

        # tensorize the source tree and reserve tree
        reserve_branches = [branch for i, branch in enumerate(branches) if predict[i] == 1]
        reserve_atoms = set()
        reserve_atoms.add(center)
        for branch in reserve_branches:
            reserve_atoms.update(branch)
        reserve_atoms = list(reserve_atoms)

        source_mol_tree = generate_tree(smiles, center, assm=False)
        reserve_mol_tree = generate_tree(smiles, center, atoms=reserve_atoms, assm=False)
        # x_batch = tensorize_tree([source_mol_tree], self.vocab, assm=False)
        reserve_x_batch = tensorize_tree([reserve_mol_tree], self.vocab, assm=False)

        # generate branch
        set_batch_nodeID([source_mol_tree], self.vocab)
        jtenc_holder, _ = JTNNEncoder.tensorize([source_mol_tree])
        x_tree_vecs, _ = self.jtnn(*jtenc_holder)
        # x_tree_vecs, _, x_mol_vecs = self.encode(x_batch[1], x_batch[2])
        # fuse tree noise
        x_tree_vecs = self.fuse_tree_noise(x_tree_vecs)
        assert x_tree_vecs.size(0) == x_mol_vecs.size(0)

        reserve_mol_tree_batch = reserve_x_batch[0]
        reserve_x_tree_vecs, _, reserve_x_mol_vecs = self.encode(reserve_x_batch[1], reserve_x_batch[2])
        assert reserve_x_tree_vecs.size(0) == reserve_x_mol_vecs.size(0)

        reserve_smiles = reserve_mol_tree_batch[0].smiles
        reserve_center = reserve_mol_tree_batch[0].center
        reserve_atoms = reserve_mol_tree_batch[0].atoms

        pred_root, pred_nodes = self.decode_tree(x_tree_vecs[0].unsqueeze(0),
                                                 x_mol_vecs[0].unsqueeze(0),
                                                 reserve_x_tree_vecs[0].unsqueeze(0),
                                                 reserve_x_mol_vecs[0].unsqueeze(0),
                                                 reserve_mol_tree_batch[0].nodes[0].wid,
                                                 reserve_smiles, reserve_center, reserve_atoms)
        gen_smiles = self.decode_smiles_from_tree(pred_root, pred_nodes, x_mol_vecs[0].unsqueeze(0),
                                                  reserve_x_mol_vecs[0].unsqueeze(0),
                                                  reserve_smiles, reserve_center, reserve_atoms)

        # TODO: fix
        if gen_smiles is None:
            return None

        combine_smile = self.combine_mol(gen_smiles, reserve_smiles, reserve_center, reserve_atoms)

        return combine_smile

    def pipeline_predict_result_with_all_actions(self, smiles):
        # predict center
        mol = get_mol(smiles)
        atom_num = mol.GetNumAtoms()
        atoms = list(range(atom_num))
        mpn_holder = MPN.tensorize([smiles], [atoms])
        x_mol_vecs = self.mpn(*mpn_holder)
        # fuse mol noise
        x_mol_vecs = self.fuse_mol_noise(x_mol_vecs)
        scores = self.center_predictor.get_predict_score(x_mol_vecs, [atom_num])
        center = scores[0].argmax().item()
        probs = F.softmax(scores[0], dim=0)
        center_prob = probs.max().item()

        # predict branch
        branches = get_branches(mol, center)
        predict = self.branch_predictor.predict(x_mol_vecs, [center], [branches])
        predict = [int(item >= 0.5) for item in predict[0]]

        # tensorize the source tree and reserve tree
        reserve_branches = [branch for i, branch in enumerate(branches) if predict[i] == 1]
        reserve_atoms = set()
        reserve_atoms.add(center)
        for branch in reserve_branches:
            reserve_atoms.update(branch)
        reserve_atoms = list(reserve_atoms)

        source_mol_tree = generate_tree(smiles, center, assm=False)
        reserve_mol_tree = generate_tree(smiles, center, atoms=reserve_atoms, assm=False)
        # x_batch = tensorize_tree([source_mol_tree], self.vocab, assm=False)
        reserve_x_batch = tensorize_tree([reserve_mol_tree], self.vocab, assm=False)

        # generate branch
        set_batch_nodeID([source_mol_tree], self.vocab)
        jtenc_holder, _ = JTNNEncoder.tensorize([source_mol_tree])
        x_tree_vecs, _ = self.jtnn(*jtenc_holder)
        # x_tree_vecs, _, x_mol_vecs = self.encode(x_batch[1], x_batch[2])
        # fuse tree noise
        x_tree_vecs = self.fuse_tree_noise(x_tree_vecs)
        assert x_tree_vecs.size(0) == x_mol_vecs.size(0)

        reserve_mol_tree_batch = reserve_x_batch[0]
        reserve_x_tree_vecs, _, reserve_x_mol_vecs = self.encode(reserve_x_batch[1], reserve_x_batch[2])
        assert reserve_x_tree_vecs.size(0) == reserve_x_mol_vecs.size(0)

        reserve_smiles = reserve_mol_tree_batch[0].smiles
        reserve_center = reserve_mol_tree_batch[0].center
        reserve_atoms = reserve_mol_tree_batch[0].atoms

        pred_root, pred_nodes = self.decode_tree(x_tree_vecs[0].unsqueeze(0),
                                                 x_mol_vecs[0].unsqueeze(0),
                                                 reserve_x_tree_vecs[0].unsqueeze(0),
                                                 reserve_x_mol_vecs[0].unsqueeze(0),
                                                 reserve_mol_tree_batch[0].nodes[0].wid,
                                                 reserve_smiles, reserve_center, reserve_atoms)
        gen_smiles = self.decode_smiles_from_tree(pred_root, pred_nodes, x_mol_vecs[0].unsqueeze(0),
                                                  reserve_x_mol_vecs[0].unsqueeze(0),
                                                  reserve_smiles, reserve_center, reserve_atoms)

        # TODO: fix
        if gen_smiles is None:
            return None, None, reserve_smiles, reserve_center, reserve_atoms, center_prob

        combine_smiles = self.combine_mol_with_root_at_first(gen_smiles, reserve_smiles, reserve_center, reserve_atoms)

        return combine_smiles, gen_smiles, reserve_smiles, reserve_center, reserve_atoms, center_prob

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, reserve_x_mol_vecs, y_tree_mess):
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess)

        x_mol_vecs = x_mol_vecs.sum(dim=1)  # average pooling?
        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)

        reserve_x_mol_vecs = reserve_x_mol_vecs.sum(dim=1)
        reserve_x_mol_vecs = reserve_x_mol_vecs.index_select(0, batch_idx)

        input_vecs = torch.cat([x_mol_vecs, reserve_x_mol_vecs], dim=1)
        input_vecs = self.A_assm(input_vecs)  # bilinear
        scores = torch.bmm(
            input_vecs.unsqueeze(1),
            cand_vecs.unsqueeze(-1)
        ).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode_tree(self, x_tree_vecs, x_mol_vecs, reserve_x_tree_vecs, reserve_x_mol_vecs,
                    root_wid, reserve_smiles, reserve_center, reserve_atoms):
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1
        assert reserve_x_tree_vecs.size(0) == 1 and reserve_x_mol_vecs.size(0) == 1

        # count extra atoms and bonds
        extra_neighbor_atom_symbols = []
        extra_neighbor_bonds = []
        reserve_mol = Chem.MolFromSmiles(reserve_smiles)
        Chem.Kekulize(reserve_mol)
        for bond in reserve_mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            bt = bond.GetBondType()
            if a1.GetIdx() not in reserve_atoms or a2.GetIdx() not in reserve_atoms:
                continue
            if a1.GetIdx() == reserve_center:
                extra_neighbor_atom_symbols.append(a2.GetSymbol())
                extra_neighbor_bonds.append(bt)
            elif a2.GetIdx() == reserve_center:
                extra_neighbor_atom_symbols.append(a1.GetSymbol())
                extra_neighbor_bonds.append(bt)

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, x_mol_vecs, reserve_x_tree_vecs, reserve_x_mol_vecs,
                                                    root_wid, extra_neighbor_atom_symbols, extra_neighbor_bonds)

        return pred_root, pred_nodes

    def decode_smiles_from_tree(self, pred_root, pred_nodes, x_mol_vecs, reserve_x_mol_vecs,
                                reserve_smiles, reserve_center, reserve_atoms):
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict)  # Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vec_pooled = x_mol_vecs.sum(dim=1)  # average pooling?
        reserve_x_mol_vec_pooled = reserve_x_mol_vecs.sum(dim=1)

        input_vecs = torch.cat([x_mol_vec_pooled, reserve_x_mol_vec_pooled], dim=1)
        input_vec = self.A_assm(input_vecs).squeeze()  # bilinear

        cur_mol = copy_edit_mol(pred_root.mol)

        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        # count extra atoms and bonds
        extra_neighbor_atom_symbols = []
        extra_neighbor_bonds = []
        reserve_mol = Chem.MolFromSmiles(reserve_smiles)
        Chem.Kekulize(reserve_mol)
        for bond in reserve_mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            bt = bond.GetBondType()
            if a1.GetIdx() not in reserve_atoms or a2.GetIdx() not in reserve_atoms:
                continue
            if a1.GetIdx() == reserve_center:
                extra_neighbor_atom_symbols.append(a2.GetSymbol())
                extra_neighbor_bonds.append(bt)
            elif a2.GetIdx() == reserve_center:
                extra_neighbor_atom_symbols.append(a1.GetSymbol())
                extra_neighbor_bonds.append(bt)

        cur_mol = self.dfs_assemble(tree_mess, input_vec, pred_nodes, cur_mol, global_amap, [], pred_root, None,
                                    extra_neighbor_atom_symbols, extra_neighbor_bonds)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)

        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol, rootedAtAtom=0))
        return Chem.MolToSmiles(cur_mol, rootedAtAtom=0) if cur_mol is not None else None

    def combine_mol(self, new_smiles, reserve_smiles, reserve_center, reserve_atoms):
        new_mol = Chem.MolFromSmiles(new_smiles)
        reserve_mol = Chem.MolFromSmiles(reserve_smiles)
        Chem.Kekulize(new_mol)
        Chem.Kekulize(reserve_mol)
        combine_mol = Chem.RWMol(Chem.MolFromSmiles(''))

        for atom in new_mol.GetAtoms():
            new_atom = copy_atom(atom)
            combine_mol.AddAtom(new_atom)
        for bond in new_mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            combine_mol.AddBond(a1, a2, bt)

        n_atoms = combine_mol.GetNumAtoms()
        atom_map = dict()

        for atom in reserve_mol.GetAtoms():
            if atom.GetIdx() not in reserve_atoms:
                continue
            if atom.GetIdx() == reserve_center:
                continue
            atom_map[atom.GetIdx()] = n_atoms
            n_atoms += 1

            new_atom = copy_atom(atom)
            combine_mol.AddAtom(new_atom)

        atom_map[reserve_center] = 0

        for bond in reserve_mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            if a1 not in reserve_atoms or a2 not in reserve_atoms:
                continue
            combine_mol.AddBond(atom_map[a1], atom_map[a2], bt)

        combine_mol = combine_mol.GetMol()
        combine_mol = Chem.MolFromSmiles(Chem.MolToSmiles(combine_mol, canonical=True))

        return Chem.MolToSmiles(combine_mol) if combine_mol is not None else None

    def combine_mol_with_root_at_first(self, new_smiles, reserve_smiles, reserve_center, reserve_atoms):
        new_mol = Chem.MolFromSmiles(new_smiles)
        reserve_mol = Chem.MolFromSmiles(reserve_smiles)
        Chem.Kekulize(new_mol)
        Chem.Kekulize(reserve_mol)
        combine_mol = Chem.RWMol(Chem.MolFromSmiles(''))

        for atom in new_mol.GetAtoms():
            new_atom = copy_atom(atom)
            combine_mol.AddAtom(new_atom)
        for bond in new_mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            combine_mol.AddBond(a1, a2, bt)

        n_atoms = combine_mol.GetNumAtoms()
        atom_map = dict()

        for atom in reserve_mol.GetAtoms():
            if atom.GetIdx() not in reserve_atoms:
                continue
            if atom.GetIdx() == reserve_center:
                continue
            atom_map[atom.GetIdx()] = n_atoms
            n_atoms += 1

            new_atom = copy_atom(atom)
            combine_mol.AddAtom(new_atom)

        atom_map[reserve_center] = 0

        for bond in reserve_mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            if a1 not in reserve_atoms or a2 not in reserve_atoms:
                continue
            combine_mol.AddBond(atom_map[a1], atom_map[a2], bt)

        combine_mol = combine_mol.GetMol()
        combine_mol = Chem.MolFromSmiles(Chem.MolToSmiles(combine_mol, canonical=True, rootedAtAtom=0))

        return Chem.MolToSmiles(combine_mol, rootedAtAtom=0) if combine_mol is not None else None

    def dfs_assemble(self, y_tree_mess, input_vec, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node,
                     extra_neighbor_atom_symbols=[], extra_neighbor_bonds=[]):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None

        cand_smiles, cand_amap = list(zip(*cands))
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])

        scores = torch.mv(cand_vecs, input_vec)
        _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        # for i in xrange(cand_idx.numel()):
        for i in range(min(cand_idx.numel(), 5)):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)  # father is already attached

            # new_mol = cur_mol.GetMol()
            # TODO: zyj
            new_mol = Chem.RWMol(cur_mol)
            atom_num = new_mol.GetNumAtoms()
            for i, (symbol, bt) in enumerate(zip(extra_neighbor_atom_symbols, extra_neighbor_bonds)):
                atom = Chem.Atom(symbol)
                new_mol.AddAtom(atom)
                new_mol.AddBond(0, atom_num+i, bt)
            new_mol = new_mol.GetMol()

            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf: continue
                cur_mol = self.dfs_assemble(y_tree_mess, input_vec, all_nodes, cur_mol, new_global_amap,
                                            pred_amap, nei_node, cur_node,
                                            extra_neighbor_atom_symbols, extra_neighbor_bonds)
                if cur_mol is None:
                    result = False
                    break
            if result:
                return cur_mol

        return None


def set_batch_nodeID(mol_tree_batch, vocab):
    tot = 0
    for mol_tree in mol_tree_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


def tensorize_tree(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    atoms_batch = [tree.atoms for tree in tree_batch]
    # 树信息 (fnode, fmess, node_graph, mess_graph, scope), mess_dict
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    # 分子图信息 (fatoms, fbonds, agraph, bgraph, scope)
    mpn_holder = MPN.tensorize(smiles_batch, atoms_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    # 考虑所有候选组合情况的图信息 （candidate 候选）
    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    # 记录属于哪棵junction tree
    batch_idx = torch.tensor(batch_idx, dtype=torch.long)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)

