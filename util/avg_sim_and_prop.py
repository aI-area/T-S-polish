#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2020/04/11 17:40:28

import logging
import numpy as np


def count_avg_sim_and_prop(data, num_decode, prop_name):
    logger = logging.getLogger('logger')

    data = [line.split() for line in data]
    data = [(a, b, float(c), float(d)) for a, b, c, d in data]
    assert len(data) % num_decode == 0

    # count similarity and property separately
    all_sim = [sim for _, _, sim, _ in data]
    all_prop = [prop for _, _, _, prop in data]
    all_sim = np.array(all_sim)
    all_prop = np.array(all_prop)
    logger.info('Average Similarity: %f, %s property: %f'
                % (float(np.mean(all_sim)), prop_name, float(np.mean(all_prop))))
