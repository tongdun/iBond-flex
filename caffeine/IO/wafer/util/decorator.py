"""
 Created by liwei on 2020/12/10.
"""
import time
from functools import wraps

from caffeine.IO.wafer.util.xlog import logger


def _calculate_cost_time(start_time):
    """
    计算耗时
    :return:
    """
    end_time = time.time()
    cost_time = end_time - start_time
    if cost_time < 60:
        return "耗时:{}秒".format(round(cost_time, 3))
    elif cost_time < 3600:
        minute_num = cost_time // 60
        second_num = cost_time % 60
        return "耗时:{}分 {}秒".format(minute_num, round(second_num, 3))
    else:
        hour_num = cost_time // (60 * 60)
        minute_left_time = cost_time - hour_num * 60 * 60
        minute_num = minute_left_time // 60
        second_left_time = cost_time - minute_left_time - minute_num * 60
        return "耗时:{}时 {}分 {}秒".format(hour_num, minute_num, round(second_left_time, 3))


def timeit(start_msg="", end_msg=""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if start_msg:
                logger.info(f"{start_msg}")

            start = time.time()
            data = func(*args, **kwargs)

            if end_msg:
                logger.info(f"{end_msg} {_calculate_cost_time(start)}")
            else:
                logger.info(f"{func.__name__} {_calculate_cost_time(start)}")
            return data

        return wrapper

    return decorator


def mointor_memory_using(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os

        logger.info(
            "当前进程的内存使用：%.4f GB"
            % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
        )
        return func(*args, **kwargs)


def arbiter(do_nothing=False, handle_func=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwds):
            if getattr(self, "is_arbiter", None):
                if do_nothing:
                    return
                if handle_func is not None:
                    _handle_func = getattr(self, handle_func, None)
                    if _handle_func is None:
                        raise RuntimeError(f"handle_func【{handle_func}】is not exist")
                    return _handle_func(*args, **kwds)
            return func(self, *args, **kwds)

        return wrapper

    return decorator


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
