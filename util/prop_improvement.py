#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2020/04/11 17:22:26

import logging
import numpy as np


def count_prop_improvement(data, decode_num, sim_delta, prop_name, using_reverse=False):
    logger = logging.getLogger('logger')

    logger.info('==========================================================================================')
    logger.info('%s PROP IMPROVEMENT: SIM DELTA-%f' % (prop_name, sim_delta))
    if using_reverse:
        logger.info('Using reverse mode! ')

    data = [line.split() for line in data]
    data = [(a, b, float(c), float(f)) for a, b, c, d, e, f in data]
    n_mols = len(data) / decode_num
    assert len(data) % decode_num == 0

    all_prop = []

    for i in range(0, len(data), decode_num):
        set_x = set([x[0] for x in data[i:i + decode_num]])
        assert len(set_x) == 1

        good = [(sim, prop) for _, _, sim, prop in data[i:i + decode_num] if 1 > sim >= sim_delta]
        if len(good) > 0:
            if not using_reverse:
                sim, prop = max(good, key=lambda x: x[1])
                all_prop.append(max(0, prop))
            else:
                sim, prop = min(good, key=lambda x: x[1])
                all_prop.append(min(0, prop))
        else:
            all_prop.append(0.0)  # No improvement

    assert len(all_prop) == n_mols
    all_prop = np.array(all_prop)

    logger.info('Evaluated on %d samples' % (n_mols,))
    mean, std = float(np.mean(all_prop)), float(np.std(all_prop))
    logger.info('%s average improvement: %f, std: %f' % (prop_name, mean, std))

    logger.info('==========================================================================================')

    return mean
