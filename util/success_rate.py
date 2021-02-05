#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2020/04/11 17:07:38

import logging


def count_success_rate(data, decode_num, sim_delta, prop_delta, prop_name, using_reverse=False, using_prop_improve=False):
    logger = logging.getLogger('logger')

    logger.info('==========================================================================================')
    logger.info('%s SUCCESS RATE: SIM DELTA-%f, DELTA-%f' % (prop_name, sim_delta, prop_delta))
    if using_reverse:
        logger.info('Using reverse mode! ')

    data = [line.split() for line in data]
    if not using_prop_improve:
        data = [(a, b, float(c), float(e)) for a, b, c, d, e, f in data]
    else:
        data = [(a, b, float(c), float(f)) for a, b, c, d, e, f in data]
    n_mols = len(data) / decode_num
    assert len(data) % decode_num == 0

    n_succ = 0.0
    for i in range(0, len(data), decode_num):
        set_x = set([x[0] for x in data[i:i + decode_num]])
        assert len(set_x) == 1

        if not using_reverse:
            good = [(sim, prop) for _, _, sim, prop in data[i:i + decode_num] if
                    1 > sim >= sim_delta and prop >= prop_delta]
        else:
            good = [(sim, prop) for _, _, sim, prop in data[i:i + decode_num] if
                    1 > sim >= sim_delta and prop <= prop_delta]
        if len(good) > 0:
            n_succ += 1

    logger.info('Evaluated on %d samples' % (n_mols,))
    success_rate = n_succ / n_mols
    logger.info('%s success rate %f' % (prop_name, success_rate))

    logger.info('==========================================================================================')

    return success_rate
