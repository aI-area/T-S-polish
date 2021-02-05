#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/12/26 14:33:29

from util.nnutils import *

from branch_jtnn.mpn import MPN


class BranchPredictor(nn.Module):
    def __init__(self, hidden_size):
        super(BranchPredictor, self).__init__()

        self.hidden_size = hidden_size

        # # branch vecs
        # self.branch_AT = Attention(args.hidden_size, args.hidden_size, args.hidden_size, args.hidden_size)
        #
        # # reserve vecs
        # self.reserve_AT = Attention(args.hidden_size, args.hidden_size, args.hidden_size, args.hidden_size)

        # FC
        self.W_o = nn.Linear(3 * hidden_size, 1)

        #  Loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, mol_vecs, centers, branches, matches):
        centers_vecs = []
        branches_vecs = []
        reserve_vecs = []
        targets = []
        batch_size = mol_vecs.shape[0]
        for i in range(batch_size):
            cur_vecs = mol_vecs[i]

            # center_vecs
            center_vec = cur_vecs[centers[i]]
            centers_vecs.extend([center_vec] * len(branches[i]))

            # branch_vecs
            cur_branches_vecs = []
            for branch in branches[i]:
                branch = torch.tensor(branch, dtype=torch.long)
                branch = create_var(branch)
                branch_vecs = cur_vecs.index_select(0, branch)
                # branch_vec = self.branch_AT(branch_vecs.mean(dim=0), branch_vecs, branch_vecs).reshape(-1)
                branch_vec = branch_vecs.mean(dim=0)
                cur_branches_vecs.append(branch_vec)
            branches_vecs.extend(cur_branches_vecs)

            # reserve vecs
            torch.zeros(self.hidden_size, dtype=torch.float32)
            reserve_index = []
            for index, match in enumerate(matches[i]):
                if len(reserve_index) == 0:
                    reserve_vecs.append(create_var(torch.zeros(self.hidden_size, dtype=torch.float32)))
                else:
                    select_vecs = [cur_branches_vecs[idx] for idx in reserve_index]
                    # select_vecs = torch.stack(select_vecs, dim=0)
                    reserve_vecs.append(torch.stack(select_vecs, dim=0).mean(dim=0))

                if match:
                    reserve_index.append(index)

            # targets
            targets.extend(matches[i])

        centers_vecs = torch.stack(centers_vecs, dim=0)
        branches_vecs = torch.stack(branches_vecs, dim=0)
        reserve_vecs = torch.stack(reserve_vecs, dim=0)
        targets = create_var(torch.tensor(targets, dtype=torch.float32))

        input_vecs = torch.cat([centers_vecs, branches_vecs, reserve_vecs], dim=1)
        predicts = self.W_o(input_vecs).reshape(-1)

        # count loss
        loss = self.bce_loss(predicts, targets)

        # count accuracy
        reserves = torch.ge(predicts, 0).int()
        targets = targets.int()
        tp = sum([1 for i in range(targets.shape[0]) if targets[i] == 1 and reserves[i] == 1])
        fp = sum([1 for i in range(targets.shape[0]) if targets[i] == 0 and reserves[i] == 1])
        fn = sum([1 for i in range(targets.shape[0]) if targets[i] == 1 and reserves[i] == 0])
        tn = sum([1 for i in range(targets.shape[0]) if targets[i] == 0 and reserves[i] == 0])

        return loss, tp, fp, fn, tn

    # batch predict
    def predict(self, mol_vecs, centers, branches):
        batch_size = mol_vecs.shape[0]
        max_len = 0
        for item in branches:
            max_len = len(item) if len(item) > max_len else max_len

        batch_reserve_vecs = [[] for i in range(batch_size)]
        batch_predicts = [[] for i in range(batch_size)]
        for step in range(max_len):
            batch_index_list = []

            centers_vecs = []
            branches_vecs = []
            reserve_vecs = []
            for i in range(batch_size):
                if len(branches[i]) <= step:
                    continue

                batch_index_list.append(i)

                cur_vecs = mol_vecs[i]

                # center_vecs
                center_vec = cur_vecs[centers[i]]
                centers_vecs.extend([center_vec])

                # branch_vecs
                branch = torch.tensor(branches[i][step], dtype=torch.long)
                branch = create_var(branch)
                branch_vecs = cur_vecs.index_select(dim=0, index=branch)
                # branch_vec = self.branch_AT(branch_vecs.mean(dim=0), branch_vecs, branch_vecs).reshape(-1)
                branch_vec = branch_vecs.mean(dim=0)
                branches_vecs.append(branch_vec)

                # reserve vecs
                if len(batch_reserve_vecs[i]) == 0:
                    reserve_vecs.append(create_var(torch.zeros(self.hidden_size, dtype=torch.float32)))
                else:
                    reserve_vecs.append(torch.stack(batch_reserve_vecs[i], dim=0).mean(dim=0))

            centers_vecs = torch.stack(centers_vecs, dim=0)
            branches_vecs = torch.stack(branches_vecs, dim=0)
            reserve_vecs = torch.stack(reserve_vecs, dim=0)

            input_vecs = torch.cat([centers_vecs, branches_vecs, reserve_vecs], dim=1)
            predicts = self.W_o(input_vecs).reshape(-1)
            predicts = F.sigmoid(predicts)
            # predicts = torch.ge(predicts, 0).int()

            for i, batch_index in enumerate(batch_index_list):
                predict = predicts[i].item()
                batch_predicts[batch_index].append(predict)

                if predict >= 0.5:
                    batch_reserve_vecs[batch_index].append(branches_vecs[i])

        return batch_predicts
