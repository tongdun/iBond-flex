#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name:
#
#  Create date:  2022/01/05

from logging import getLogger
from caffeine.utils.convergence import converge_func_factory

class TestConvergeFunciton(object):
    def __init__(self, eps, max_iter_num, loss_decay_rate):
        self.logger = getLogger(self.__class__.__name__)
        self.eps = eps
        self.max_iter_num = max_iter_num
        self.loss_decay_rate = loss_decay_rate

    def test_convergefunction(self, current_loss, early_stop_method):
        converge_func = converge_func_factory(early_stop_method=early_stop_method, tol=self.eps)
        iter_num = 0
        while iter_num < self.max_iter_num:
            current_loss *= self.loss_decay_rate
            converge_flag = converge_func.is_converge(current_loss)
            if converge_flag:
                self.logger.info(f'{early_stop_method}_converge_iter:{iter_num}')
                break
            iter_num += 1

if __name__ == '__main__':
    eps = 1e-4
    max_iter_num = 500
    loss_decay_rate = 0.25
    test= TestConvergeFunciton(eps, max_iter_num, loss_decay_rate)
    current_loss = 50
    test.test_convergefunction(current_loss, early_stop_method='diff')
    test.test_convergefunction(current_loss, early_stop_method='abs')