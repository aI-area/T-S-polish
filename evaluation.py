#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/12/02 15:04:55

import argparse
import logging
import sys
import json
from datetime import datetime
from tqdm import tqdm

import rdkit
import torch

from util import LOG
from util.logp_score import evaluation_logp04, evaluation_logp06, evaluation_reverse_logp04
from util.qed_score import evaluation_qed
from util.drd2_score import evaluation_drd2
from util.multi_prop_score import evaluation_multi_prop

from branch_jtnn.branch_jtnn import BranchJTNN
from branch_jtnn.mol_tree import Vocab


def program_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', required=True)
    parser.add_argument('--load_model_dir', required=True)
    parser.add_argument('--load_model_config_dir', required=True)

    parser.add_argument('--task_tag', required=True, choices=['drd2', 'qed', 'logp04', 'logp06',
                                                              'GuacaMol_qed', 'Moses_qed',
                                                              'GuacaMol_multi_prop', 'reverse_logp04'])
    parser.add_argument('--metric_type', required=True, choices=['M1', 'M2', 'M3', 'M4', 'M5', 'M6'])
    parser.add_argument('--decode_num', type=int, default=1)

    parser.add_argument('--seed', type=int, default=4)

    # device setting
    device_ids = list(range(torch.cuda.device_count()))
    parser.add_argument('--device', type=int, choices=device_ids, required=False, default=0)

    args = parser.parse_args()

    # read the model config data
    with open(args.load_model_config_dir, 'r') as f:
        config_data = json.load(f)
    args.vocab = config_data['vocab']
    args.hidden_size = config_data['hidden_size']
    args.rand_size = config_data['rand_size']
    args.depthT = config_data['depthT']
    args.depthG = config_data['depthG']
    args.share_embedding = config_data['share_embedding']
    args.use_molatt = config_data['use_molatt']

    return args


def initialize(args):
    current_time = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())
    sys.setrecursionlimit(10000)
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    arg_info = '%s_%s_HS_%d_RS_%d' % (
        args.task_tag,
        args.metric_type,
        args.hidden_size,
        args.rand_size
    )
    LOG.init(file_name=current_time + '_Evaluation' + '_' + arg_info)
    logger = logging.getLogger('logger')
    logger.info(args)

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)

    return logger


def evaluation(args, logger):
    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    model = BranchJTNN(vocab, args).cuda()
    logger.info("Load model: %s" % args.load_model_dir)
    model.load_state_dict(torch.load(args.load_model_dir, map_location=lambda storage, loc: storage.cuda(args.device)))
    logger.info("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    model.eval()

    with open(args.eval_file, 'r') as f:
        eval_data = f.readlines()
    eval_data = [line.strip().split()[0] for line in eval_data]

    result = []

    for it, source_smiles in enumerate(tqdm(eval_data)):
        for i in range(args.decode_num):
            try:
                with torch.no_grad():
                    predict_smiles = model.pipeline_predict(source_smiles)
                    result.append('%s %s' % (source_smiles, predict_smiles))
            except Exception as e:
                logger.error(e)
                result.append('%s %s' % (source_smiles, 'None'))
                continue

    if args.task_tag == 'drd2':
        evaluation_drd2(result, args.metric_type, decode_num=args.decode_num)
    elif args.task_tag == 'qed' or args.task_tag == 'GuacaMol_qed' or args.task_tag == 'Moses_qed':
        evaluation_qed(result, args.metric_type, decode_num=args.decode_num)
    elif args.task_tag == 'logp04':
        evaluation_logp04(result, args.metric_type, decode_num=args.decode_num)
    elif args.task_tag == 'logp06':
        evaluation_logp06(result, args.metric_type, decode_num=args.decode_num)
    elif args.task_tag == 'reverse_logp04':
        evaluation_reverse_logp04(result, args.metric_type, decode_num=args.decode_num)
    elif args.task_tag == 'GuacaMol_multi_prop':
        evaluation_multi_prop(result, args.metric_type, decode_num=args.decode_num)


if __name__ == '__main__':
    args = program_config()

    logger = initialize(args)
    evaluation(args, logger)
