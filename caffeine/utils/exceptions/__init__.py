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
#  File name: exceptions
#
#  Create date: 2020/12/14
#
import json
import os
from logging import getLogger


class IBondException(BaseException):
    """
    IbondException is a excpetion for raise exception to middleware.

    Json file format:
    {
        "IBondException": {
            "name": "ibond_error",
            "code": "exp_ibond"
        }
    }

    Returns:
     None
    Example:
    >>> from exceptions import *
    >>> try:
    >>>     raise IBondException
    >>> except IBondException as error:
    >>>     error.report()
    >>> # Output:
    >>> Error name is ibond_error, error code is exp_ibond.'
    """

    name = 'ibond_error'
    code = 'exp_ibond'

    # todo We need middleware to report the exception. This part will be imported

    @classmethod
    def report(cls, reporter=None):  # todo type hint
        if not reporter:
            logger = getLogger(cls.__name__)
            logger.info(f' Error name is {cls.name}, error code is {cls.code}.')
        else:
            print(f'错误名称: {cls.name} 错误代码：{cls.code}')
    # todo 将信息推送到中间件接口，然后前端输出
    # self.reporter(cls.name, cls.code)


json_path = os.path.dirname(__file__)
all_filenames = os.listdir(json_path)
filenames = [i for i in all_filenames if os.path.splitext(i)[1] == '.json']

exception_config = {}
for filename in filenames:
    with open(os.path.join(json_path, filename), 'r') as f:
        exception_config.update(json.load(f))

# TODO: change locals to list
for exp, attr in exception_config.items():
    locals()[exp] = type(exp, (IBondException,), {'name': attr['name'], 'code': attr['code']})
