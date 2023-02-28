#!/usr/bin/python3
#
#  _____________             _________   ___       __      ________
#  ___(_)__  __ )__________________  /   __ |     / /_____ ___  __/____________
#  __  /__  __  |  __ \_  __ \  __  /    __ | /| / /_  __ `/_  /_ _  _ \_  ___/
#  _  / _  /_/ // /_/ /  / / / /_/ /     __ |/ |/ / / /_/ /_  __/ /  __/  /
#  /_/  /_____/ \____//_/ /_/\__,_/      ____/|__/  \__,_/ /_/    \___//_/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond Wafer
#  File name: hooks.py
#  Created by liwei on 2020/12/8.

import time
from typing import Callable, Any, Dict
import json

from abc import ABC, abstractmethod
from logging import getLogger


class Hook(ABC):
    @abstractmethod
    def pre_hook(self, context: Dict[str, Any]):
        pass

    @abstractmethod
    def iter_hook(self, context: Dict[str, Any]):
        pass

    @abstractmethod
    def post_hook(self, context: Dict[str, Any]):
        pass


class Timer(object):
    def __init__(self):
        """
        A utility class to calculate ETA and elapsed_time.

        Example:
        >>> timer = Timer()
        >>> print(timer.elapsed_time)
        >>> print(timer.eta(0.5))
        """
        self._start_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """
        Time since timer is initialized.
        Args: None
        Returns:
            float, the time in seconds since the epoch as a floating point number.
        """
        return time.time() - self._start_time

    def eta(self, finished_ration: float) -> float:
        """
        Time since timer is initialized.
        Args:
            finished_ration: the ratio of Finished Tasks vs Overall Tasks.
        Returns:
            float, the time in seconds of Estimated Finished Time.
        """
        return self.elapsed_time / (finished_ration + 1e-9)


def iteration_progress(context: Dict[str, Any]):
    iteration_num = context.get("iteration_num")
    current_iteration = context.get("current_iteration")
    return current_iteration * 1.0 / iteration_num

def need_do_iteration(context: Dict[str, Any]):
    iteration_num = context.get("iteration_num")
    current_iteration = context.get("current_iteration")
    if current_iteration % max(1, int(iteration_num / 10)) == 0 or current_iteration == iteration_num:
        return True
    else:
        return False


class Time_hook(Hook):
    def __init__(self, report_callback: callable):
        self.logger = getLogger(self.__class__.__name__)
        self.timer: Timer = None
        self.report_callback = report_callback

    def pre_hook(self, context: Dict[str, Any]):
        self.timer = Timer()

    def iter_hook(self, context: Dict[str, Any]):
        try:
            if not need_do_iteration(context):
                return
            current_epoch = context.get("current_epoch")
            max_num_epoch = context.get("max_num_epoch")
            epoch_progress = current_epoch * 1.0 / max_num_epoch
            rate_of_progress = epoch_progress * iteration_progress(context)
            eta = self.timer.eta(rate_of_progress)
            self.logger.info(
                f"Reporting time estimation epoch={current_epoch}, ETA={eta:.3f} seconds."
            )
            self.report_callback("num_epoches", json.dumps(current_epoch))
            self.report_callback("eta", json.dumps(eta))
        except:
            self.logger.exception("Encounter errors during ETA reporting.")
        pass

    def post_hook(self, context: Dict[str, Any]):
        pass


class Loss_hook(Hook):
    """
    This class is used to report training loss. 
    A exponential smoothing method is applied to loss before reporting.
    """
    def __init__(self, report_callback: Callable, decay: float=0.5):
        self.logger = getLogger(self.__class__.__name__)
        self.report_callback = report_callback
        self.decay = decay
        self.last_loss = None

    def pre_hook(self, iteration_num: int):
        pass

    def iter_hook(self, context: Dict[str, Any]):
        try:
            if not need_do_iteration(context):
                return
                
            current_loss = context.pop("loss")
            current_epoch = context.get("current_epoch") - 1

            if self.last_loss is None:
                smoothed_loss = current_loss
            else:
                smoothed_loss = self.last_loss * self.decay + (1 - self.decay) * current_loss
            
            self.last_loss = smoothed_loss

            self.logger.info(
                f"Reporting loss epoch={current_epoch}, loss={smoothed_loss}."
            )
            self.report_callback(
                "loss",
                json.dumps({
                    "epoch": current_epoch,
                    "iteration": current_epoch + iteration_progress(context),
                    "loss": smoothed_loss,
                }),
            )
        except:
            self.logger.exception("Encounter errors during Loss reporting.")
        pass

    def post_hook(self, context: Dict[str, Any]):
        pass
