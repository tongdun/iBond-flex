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
#  Create date: 2020/11/24
#


from functools import wraps
from logging import Logger, getLogger
from os import path
from typing import Callable

from .init_logs import setup_logging

setup_logging()

from caffeine.utils.config import middleware_version

if middleware_version == 'light':
    from caffeine.IO.wafer.dataframe.light_dataframe import LightDataFrame as IBondDataFrame
    from caffeine.IO.wafer import Wafer as Middleware
    from caffeine.IO.wafer.session.ibond_session import IBondSession as Context
else:
    raise NotImplementedError(
        f'Middleware version {middleware_version} is not supported. Use light or mock instead.')


class FunctionAutoLog(object):
    def __init__(self, logger_name: str, mode: str = 'simple'):
        """
        FunctionAutoLog is a decorator class used to add logs to a selected function.

        Args:
            logger_name: str, name for python standard logger. __file__ is often used here.
            mode: [Optional] str, mode to choose logging output style. currently, 'simple' is the only choice.

        Returns:
            None

        Example:
        >>> from utils import FunctionAutoLog

        >>> @FunctionAutoLog(__file__)
        >>> def test():
        >>>     pass
        >>> test()

        >>> # Output:
        >>> 2020-11-26 11:29:45,562 INFO test_log.py test() started.
        >>> 2020-11-26 11:29:45,600 INFO test_log.py test() ended.
        """
        self.logger = getLogger(path.basename(logger_name))
        self.mode = mode

    def __call__(self, func: Callable) -> Callable:
        """
        This function will run once when the decorated function is called.
        """
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            self.logger.debug(f'{func.__name__}() started.')
            x = func(*args, **kwargs)
            self.logger.debug(f'{func.__name__}() ended.')
            return x
        return wrapped_function


class ClassMethodAutoLog(object):
    def __init__(self, logger: Logger = None, mode: str = 'simple'):
        """
        ClassMethodAutoLog is a decorator class used to add logs to a selected class method.

        Args:
            logger: Logger, python standard logger
            mode: [Optional] str, mode to choose logging output style. currently, 'simple' is the only choice.

        Returns:
            None

        Example:
        >>> from utils import ClassMetohdAutoLog
        >>> class A():
        >>>     @ClassMetohdAutoLog()
        >>>     def test(self):
        >>>         pass

        >>> A().test()
        >>> # Output:
        >>> 2020-11-26 11:29:45,562 INFO A test() started.
        >>> 2020-11-26 11:29:45,600 INFO A test() ended.
        """
        self.logger = logger
        self.mode = mode

    def __call__(self, func: Callable) -> Callable:
        """
        This function will run once when the decorated function is called.
        """
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            if self.logger is None:
                self.logger = getLogger(args[0].__class__.__name__)
            self.logger.debug(f'{func.__name__}() started.')
            x = func(*args, **kwargs)
            self.logger.debug(f'{func.__name__}() ended.')
            return x
        return wrapped_function
