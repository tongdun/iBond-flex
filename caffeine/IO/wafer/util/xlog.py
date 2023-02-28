"""
 Created by liwei on 2020/12/10.
"""
import logging


class XLog(object):
    """
    log
    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_singleton"):
            cls._singleton = super().__new__(cls, *args, **kwargs)
        return cls._singleton

    def __init__(self, logger=None, task_id=None, task_type=None):
        if not logger:
            formatter = logging.Formatter(
                "%(asctime)s %(funcName)s %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            logger = logging.getLogger("spark_log")
            logger.setLevel(logging.INFO)
            # 控制台日志
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            logger.addHandler(console)
        self.logger = logger
        self.task_id = task_id
        self.task_type = task_type

    def info(self, msg, *args, **kwargs):
        if 1 == self.task_type:
            self.logger.info(self.task_id, msg, *args, **kwargs)
        else:
            self.logger.info(msg)

    def debug(self, msg, *args, **kwargs):
        if 1 == self.task_type:
            self.logger.debug(self.task_id, msg, *args, **kwargs)
        else:
            self.logger.debug(msg)

    def warn(self, msg, *args, **kwargs):
        self.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if 1 == self.task_type:
            self.logger.warning(self.task_id, msg, *args, **kwargs)
        else:
            self.logger.warning(msg)

    def error(self, msg, *args, **kwargs):
        if 1 == self.task_type:
            self.logger.error(self.task_id, msg, *args, **kwargs)
        else:
            self.logger.error(msg)


logger = XLog()
