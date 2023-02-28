#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from functools import wraps
from logging import Logger, getLogger
from typing import Callable
from os import path

from .init_logs import setup_logging

setup_logging()


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
            self.logger.info(f'{func.__name__}() started.')
            x = func(*args, **kwargs)
            self.logger.info(f'{func.__name__}() ended.')
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
            self.logger.info(f'{func.__name__}() started.')
            x = func(*args, **kwargs)
            self.logger.info(f'{func.__name__}() ended.')
            return x
        return wrapped_function
