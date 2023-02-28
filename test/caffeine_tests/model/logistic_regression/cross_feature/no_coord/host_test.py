#!/usr/bin/python3
#
#  Copyright 2021 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
# -*- coding: utf-8 -*-
# @Time    : 2022/11/30 19:23
# @Author  : iBond Authors
# @File    : host_test.py
# -*- File description -*-
# -*- File description -*-
import time
from caffeine_tests.model.hetero_mainflow import case_template_no_coord, run
from common_params import test_cases, test_case
##############################################################################
# ray.init()
# @profile(precision=4, stream=open('./memory_guest_profiler.log','w+'))
def test():
    s = time.time()
    casename = "test_open_source"

    run(casename, test_case['host']['input'])
    e = time.time()
    ss = e - s
    m, s = divmod(ss, 60)
    h, m = divmod(m, 60)
    print("run time is %d:%02d:%02d" % (h, m, s))
test()
