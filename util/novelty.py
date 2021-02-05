#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2020/04/11 17:53:09

import logging


def count_novelty(data, target_file):
    logger = logging.getLogger('logger')

    with open(target_file, 'r') as f:
        target = [line.strip("\r\n ") for line in f]
    target = set(target)

    data = [line.split() for line in data]
    data = [b for a, b, c, d in data]
    preds = set(data)

    x = len(preds & target)
    logger.info('Predict num: %d, target num: %d, common num: %d' % (len(preds), len(target), x))
    novelty = 1 - x * 1.0 / len(target)
    logger.info('Novelty: %f' % novelty)

    return novelty


