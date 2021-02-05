#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/10/05 10:59:55

import logging.config
from datetime import datetime
import os


def init(log_dir='saved/log', file_name='{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())):
    logger = logging.getLogger('logger')
    logger.propagate = False

    console_handler = logging.StreamHandler()

    if file_name is not None:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        file_name = os.path.join(log_dir, file_name + '.log')
        file_handler = logging.FileHandler(filename=file_name, mode='w')

    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    if file_name is not None:
        file_handler.setLevel(logging.DEBUG)

    # formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(module)s - %(lineno)04d | %(message)s')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(module)s-%(lineno)04d | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    if file_name is not None:
        file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    if file_name is not None:
        logger.addHandler(file_handler)


