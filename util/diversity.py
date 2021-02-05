#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2020/04/12 18:01:57

import logging
import numpy as np

from props import *


def count_diversity(data, num_decode, sim_delta, prop_delta):
    logger = logging.getLogger('logger')

    data = [line.split() for line in data]
    data = [(a, b, float(c), float(d)) for a, b, c, d in data]
    assert len(data) % num_decode == 0

    def convert(x):
        return None if x == "None" else x

    all_div = []
    n_succ = 0
    for i in range(0, len(data), num_decode):
        set_x = set([x[0] for x in data[i:i + num_decode]])
        assert len(set_x) == 1

        good = [convert(y) for x, y, sim, prop in data[i:i + num_decode] if
                sim >= sim_delta and prop >= prop_delta]
        if len(good) == 0:
            continue

        good = list(set(good))
        if len(good) == 1:
            all_div.append(0.0)
            continue
        n_succ += 1

        div = 0.0
        tot = 0
        for i in range(len(good)):
            for j in range(i + 1, len(good)):
                div += 1 - similarity(good[i], good[j])
                tot += 1
        div /= tot
        all_div.append(div)

    all_div = np.array(all_div)
    mean = float(np.mean(all_div))
    std = float(np.std(all_div))

    logger.info('Diversity: average %f, std %f' % (mean, std))

    return mean
