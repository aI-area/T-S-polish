#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2020/04/11 16:34:56

from props import *
from util.success_rate import count_success_rate
from util.prop_improvement import count_prop_improvement


def evaluation_drd2(data, decode_num=1):
    result = count_drd2_score(data)
    # M1
    count_success_rate(result, decode_num=decode_num,
                       sim_delta=0.3, prop_delta=0.6, prop_name='drd2',
                       using_reverse=False, using_prop_improve=False)
    # M2
    count_success_rate(result, decode_num=decode_num,
                       sim_delta=0.4, prop_delta=0.8, prop_name='drd2',
                       using_reverse=False, using_prop_improve=False)
    # M3
    count_success_rate(result, decode_num=decode_num,
                       sim_delta=0.4, prop_delta=0.5, prop_name='drd2',
                       using_reverse=False, using_prop_improve=False)
    # M4
    count_prop_improvement(result, decode_num=decode_num,
                           sim_delta=0.3, prop_name='drd2',
                           using_reverse=False)
    # M5
    count_prop_improvement(result, decode_num=decode_num,
                           sim_delta=0.4, prop_name='drd2',
                           using_reverse=False)


def count_drd2_score(data):
    result = []
    for line in data:
        x, y = line.split()
        if y == "None":
            y = None
        sim2D = similarity(x, y)
        try:
            prop_x = drd2(x)
            prop_y = drd2(y)
            result.append('%s %s %f %f %f %f' % (x, y, sim2D, prop_x, prop_y, prop_y-prop_x))
        except Exception as e:
            result.append('%s %s %f %f %f %f' % (x, y, sim2D, 0.0, 0.0, 0.0))

    return result
