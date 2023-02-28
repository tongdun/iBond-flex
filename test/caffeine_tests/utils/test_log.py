#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: test_log
#
#  Create date: 2020/11/25
#
# from init_logs import setup_logging

import os
from logging import getLogger

from caffeine.utils import FunctionAutoLog, ClassMethodAutoLog


class XGB():
    def __init__(self):
        self.logger = getLogger(self.__class__.__name__)
        self.data = ['1', '2', '3']

    def __repr__(self) -> str:
        return "-".join(self.data)

    @ClassMethodAutoLog()
    def test(self):
        self.logger.info("123")
        self.logger.error("123")
        self.logger.debug("123")
        self.logger.info(f"Status: data={self.data}")
        self.logger.info(self)


@FunctionAutoLog(__file__)
def test():
    pass

a = XGB()
a.test()
test()