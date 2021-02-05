#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/12/26 11:06:40

from util.nnutils import *

from branch_jtnn.mpn import MPN


class CenterPredictor(nn.Module):
    def __init__(self, hidden_size):
        super(CenterPredictor, self).__init__()

        # FC
        self.W_o = nn.Linear(2 * hidden_size, 1)

        #  Loss
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, mol_vecs, atom_num, targets):
        predicts = self.get_predict_score(mol_vecs, atom_num)

        max_len = max(atom_num)
        for i in range(len(predicts)):
            predicts[i] = F.pad(predicts[i], [0, max_len - atom_num[i]], value=-10000)
        predicts = torch.stack(predicts, dim=0)
        for target in targets:
            if len(target) < max_len:
                target += [-10000] * (max_len - len(target))
        targets_tensor = torch.tensor(targets, dtype=torch.float32).cuda()

        predicts = F.log_softmax(predicts, dim=1)
        softmax_targets = F.softmax(targets_tensor, dim=1)

        loss = self.kl_loss(predicts, softmax_targets)

        # count acc
        predicts_center = torch.argmax(predicts, dim=1)
        correct_num = 0
        for i, target in enumerate(targets):
            if target[predicts_center[i]] == max(target):
                correct_num += 1
        total_num = len(targets)

        return loss, correct_num, total_num

    def get_predict_score(self, mol_vecs, atom_num):
        context_vecs = []
        atom_vecs = []
        for i, num in enumerate(atom_num):
            cur_vecs = mol_vecs[i, :num]
            atom_vecs.append(cur_vecs)

            # query = self.W_Q(cur_vecs)
            # key = self.W_K(cur_vecs)
            # att = F.softmax(torch.mm(query, key.T), dim=1).unsqueeze(dim=-1)
            att = F.softmax(torch.mm(cur_vecs, cur_vecs.T), dim=1).unsqueeze(dim=-1)
            cur_vecs = cur_vecs.unsqueeze(dim=0)
            cur_vecs = (cur_vecs * att).sum(dim=1)

            context_vecs.append(cur_vecs)

        context_vecs = torch.cat(context_vecs, dim=0)
        atom_vecs = torch.cat(atom_vecs, dim=0)
        input_vecs = torch.cat([context_vecs, atom_vecs], dim=1)

        output_vecs = self.W_o(input_vecs).squeeze(-1)

        predicts = []
        st = 0
        for num in atom_num:
            predict = output_vecs[st: st + num]
            predicts.append(predict)
            st = st + num

        return predicts
