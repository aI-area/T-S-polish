#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2020/04/11 16:39:19

from props import *
from util.success_rate import count_success_rate
from util.prop_improvement import count_prop_improvement


def evaluation_qed(data, metric_type, decode_num=1):
    result = count_qed_score(data)

    if metric_type == 'M1':
        # M1
        count_success_rate(result, decode_num=decode_num,
                           sim_delta=0.3, prop_delta=0.6, prop_name='qed',
                           using_reverse=False, using_prop_improve=False)
    elif metric_type == 'M2':
        # M2
        count_success_rate(result, decode_num=decode_num,
                           sim_delta=0.4, prop_delta=0.8, prop_name='qed',
                           using_reverse=False, using_prop_improve=False)
    elif metric_type == 'M3':
        # M3
        count_success_rate(result, decode_num=decode_num,
                           sim_delta=0.4, prop_delta=0.9, prop_name='qed',
                           using_reverse=False, using_prop_improve=False)
    elif metric_type == 'M4':
        # M4
        count_prop_improvement(result, decode_num=decode_num,
                               sim_delta=0.3, prop_name='qed',
                               using_reverse=False)
    elif metric_type == 'M5':
        # M5
        count_prop_improvement(result, decode_num=decode_num,
                               sim_delta=0.4, prop_name='qed',
                               using_reverse=False)
    else:
        raise Exception('metric_type must be chosen from M1 to M5! ')


def count_qed_score(data):
    result = []
    for line in data:
        x, y = line.split()
        if y == "None":
            y = None
        sim2D = similarity(x, y)
        try:
            prop_x = qed(x)
            prop_y = qed(y)
            result.append('%s %s %f %f %f %f' % (x, y, sim2D, prop_x, prop_y, prop_y-prop_x))
        except Exception as e:
            result.append('%s %s %f %f %f %f' % (x, y, sim2D, 0.0, 0.0, 0.0))

    return result
