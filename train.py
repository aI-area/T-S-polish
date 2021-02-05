import argparse
import logging
import math
import os
import sys
from datetime import datetime
import json

import numpy as np
import rdkit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import LOG

from branch_jtnn.datautils import TrainFolder
from branch_jtnn.mol_tree import Vocab
from branch_jtnn.branch_jtnn import BranchJTNN


def program_config():
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model_save_dir', required=False, default='saved/model')
    parser.add_argument('--tensorboard_save_dir', required=False, default='saved/runs')
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--load_model_dir', required=False, default=None)

    parser.add_argument('--task_tag', required=False, choices=['drd2', 'qed', 'logp04', 'logp06',
                                                               'GuacaMol_qed', 'Moses_qed',
                                                               'GuacaMol_multi_prop', 'reverse_logp04'])

    # network structure
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--rand_size', type=int, default=8)
    parser.add_argument('--share_embedding', action='store_true')
    parser.add_argument('--use_molatt', action='store_true')
    parser.add_argument('--depthT', type=int, default=6)
    parser.add_argument('--depthG', type=int, default=3)
    parser.add_argument('--mpn_dropout_rate', type=float, default=0)

    # training hyper parameters
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_interval', type=int, default=1)

    # device setting
    device_ids = list(range(torch.cuda.device_count()))
    device_ids.append(-1)
    parser.add_argument('--device', type=int, choices=device_ids, required=False, default=0)

    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    return args


def initialize(args):
    sys.setrecursionlimit(10000)
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    arg_info = '_%s_LR_%f_HS_%d_RS_%d_AR_%.2f_AI_%d_%s' % (
        args.task_tag,
        args.lr,
        args.hidden_size,
        args.rand_size,
        args.anneal_rate,
        args.anneal_interval,
        'share_embedding' if args.share_embedding else 'not_share_embedding'
    )
    LOG.init(file_name=current_time + '_' + arg_info)
    logger = logging.getLogger('logger')
    logger.info(args)

    # create the tensorboard log saved folder
    if not os.path.isdir(args.tensorboard_save_dir):
        os.makedirs(args.tensorboard_save_dir)
    # set the tensorboard writer
    train_tb_log_dir = os.path.join(args.tensorboard_save_dir, current_time + '_' + arg_info + '_train')
    tb_suffix = '_' + arg_info
    train_writer = SummaryWriter(log_dir=train_tb_log_dir,
                                 filename_suffix=tb_suffix)

    # create the model saved folder
    if args.model_save_dir is not None:
        # args.model_save_dir = os.path.join(args.model_save_dir, current_time + '_' + arg_info)
        args.model_save_dir = os.path.join(args.model_save_dir, f'{args.task_tag}')
        if not os.path.isdir(args.model_save_dir):
            os.makedirs(args.model_save_dir)
    # save the model config
    with open(os.path.join(args.model_save_dir, 'model_config.json'), 'w') as f:
        json.dump({'vocab': args.vocab,
                   'hidden_size': args.hidden_size,
                   'rand_size': args.rand_size,
                   'share_embedding': args.share_embedding,
                   'use_molatt': args.use_molatt,
                   'depthT': args.depthT,
                   'depthG': args.depthG}, f)

    return logger, train_writer


def training(args, logger, train_writer):

    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    model = BranchJTNN(vocab, args).cuda()
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if args.load_epoch > 0 and args.load_model_dir is not None:
        logger.info("Load model: %s, Load epoch: %d" % (args.load_model_dir, args.load_epoch))
        model.load_state_dict(torch.load(args.load_model_dir,
                                         map_location=lambda storage, loc: storage.cuda(args.device)))

    # for x in model.parameters():
    #     print(x.shape)
    logger.info("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
    logger.info("Model #Train params: %dK" % (sum([x.nelement() for x in model.parameters() if x.requires_grad]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    PRINT_ITER = 1
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    for epoch in range(args.load_epoch + 1, args.epoch + 1):
        logger.info('Current epoch: %d' % epoch)
        logger.info("Learning rate: %.6f" % scheduler.get_lr()[0])

        model.train()

        loader = TrainFolder(args.train_file, vocab, args.batch_size, num_workers=8, shuffle=True)
        meters = np.zeros(10)
        loss_list, kl_div_list, total_center_correct_num, total_center_total_num, total_branch_tp, total_branch_fp, total_branch_fn, total_branch_tn, wacc_list, tacc_list, sacc_list = [], [], 0, 0, 0, 0, 0, 0, [], [], []
        for it, batch in enumerate(loader):

            x_batch, reserve_x_batch, y_batch, scores, centers, branches, matches = batch
            try:
                model.zero_grad()
                loss, kl_div, center_correct_num, center_total_num, branch_tp, branch_fp, branch_fn, branch_tn, wacc, tacc, sacc = model(x_batch, reserve_x_batch, y_batch, scores, centers, branches, matches, args.beta)
                loss.backward()
            except Exception as e:
                logger.error(e)
                continue

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, center_correct_num, center_total_num, branch_tp, branch_fp, branch_fn, branch_tn, wacc * 100, tacc * 100, sacc * 100])

            loss_list.append(loss.item())
            kl_div_list.append(kl_div)
            total_center_correct_num += center_correct_num
            total_center_total_num += center_total_num
            total_branch_tp += branch_tp
            total_branch_fp += branch_fp
            total_branch_fn += branch_fn
            total_branch_tn += branch_tn
            wacc_list.append(wacc)
            tacc_list.append(tacc)
            sacc_list.append(sacc)

            if (it + 1) % PRINT_ITER == 0:
                meters /= PRINT_ITER
                logger.info("KL: %.2f, Center: %.2f, Branch: %.2f, Branch Acc Positive: %.2f, Branch Acc Negative: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                    meters[0], meters[1]/meters[2]*100, (meters[3]+meters[6])/(meters[4]+meters[5])*100,
                    meters[3] / (meters[3] + meters[5]), meters[6] / (meters[6] + meters[4]),
                    meters[7], meters[8], meters[9], param_norm(model), grad_norm(model)))
                meters *= 0

        train_loss = np.mean(loss_list)
        train_kl_div = np.mean(kl_div_list)
        train_center_acc = total_center_correct_num / total_center_total_num
        train_branch_acc = (total_branch_tp + total_branch_tn) / \
                           (total_branch_tp + total_branch_fp + total_branch_fn + total_branch_tn)
        train_branch_acc_reserve = total_branch_tp / (total_branch_tp + total_branch_fn)
        train_branch_acc_delete = total_branch_tn / (total_branch_tn + total_branch_fp)
        train_wacc = np.mean(wacc_list)
        train_tacc = np.mean(tacc_list)
        train_sacc = np.mean(sacc_list)

        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('kl divergence', train_kl_div, epoch)
        train_writer.add_scalar('center accuracy', train_center_acc, epoch)
        train_writer.add_scalar('branch accuracy', train_branch_acc, epoch)
        train_writer.add_scalar('branch accuracy (reserve)', train_branch_acc_reserve, epoch)
        train_writer.add_scalar('branch accuracy (delete)', train_branch_acc_delete, epoch)
        train_writer.add_scalar('word accuracy', train_wacc, epoch)
        train_writer.add_scalar('topo accuracy', train_tacc, epoch)
        train_writer.add_scalar('assemble accuracy', train_sacc, epoch)
        train_writer.flush()
        logger.info('Train Loss: %.2f, KL: %.2f, Center: %.2f, Branch: %.2f, Branch Acc Reserve: %.2f, Branch Acc Delete: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f' %
                    (float(train_loss), float(train_kl_div), train_center_acc * 100,
                     train_branch_acc * 100, train_branch_acc_reserve * 100, train_branch_acc_delete * 100,
                     train_wacc * 100, train_tacc * 100, train_sacc * 100))

        # save model
        if args.model_save_dir is not None:
            torch.save(model.state_dict(), args.model_save_dir + "/model.iter-" + str(epoch))

        if epoch % args.anneal_interval == 0:
            scheduler.step()


if __name__ == '__main__':
    current_time = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())
    args = program_config()

    logger, train_writer = initialize(args)
    training(args, logger, train_writer)
