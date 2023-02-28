#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: convergence.py
#
#  Create date: 2022/01/04

from logging import getLogger
from caffeine.utils.exceptions import ArgumentError


class ConvergeFunction(object):
    def __init__(self, eps):
        self.logger = getLogger(self.__class__.__name__)
        self.eps = eps

    def is_converge(self, loss):
        pass


class DiffConverge(ConvergeFunction):
    def __init__(self, eps):
        super().__init__(eps=eps)
        self.pre_loss = None

    def is_converge(self, loss):
        self.logger.info(f'In diff converge function, pre_loss:{self.pre_loss}, current_loss:{loss}')
        converge_flag = False
        if self.pre_loss is None:
            pass
        elif abs(self.pre_loss-loss) < self.eps:
            converge_flag = True
        self.pre_loss = loss
        return converge_flag


class AbsConverge(ConvergeFunction):
    def is_converge(self, loss):
        self.logger.info(f'In abs converge function, current_loss:{loss}')
        if loss <= self.eps:
            converge_flag = True
        else:
            converge_flag = False

        return converge_flag


def converge_func_factory(early_stop_method, tol):

    if early_stop_method == 'diff':
        return DiffConverge(tol)
    elif early_stop_method == 'abs':
        return AbsConverge(tol)
    else:
        raise ArgumentError(f'Converge Function method {early_stop_method} not support yet')