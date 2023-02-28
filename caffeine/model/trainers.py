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
#  File name: trainers.py
#
#  Create date: 2020/01/13

import json
import time
from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import Optional, Callable, List, Dict
from caffeine.IO.wafer.util.hooks import Time_hook, Loss_hook

from caffeine.model.base_model import ModelInfo
from caffeine.utils import ClassMethodAutoLog
from caffeine.utils import IBondDataFrame
from caffeine.utils import config
from caffeine.utils.federal_commu import Radio, FederalInfoParser

HOOKS = {
    'time': Time_hook,
    'loss': Loss_hook
}


class Trainer(object):
    @abstractmethod
    def train(self,
              train_data: IBondDataFrame,
              val_data: Optional[IBondDataFrame] = None
              ) -> List[str]:
        raise NotImplementedError

class EpochTrainer(Trainer):
    def __init__(self,
                 max_num_epoches: int,
                 train_epoch: Callable,
                 save_model: Optional[Callable] = None,
                 predict: Optional[Callable] = None,
                 metrics: Optional[Callable] = None,
                 report_callback: Optional[Callable] = None,
                 report_hooks: List[str] = ['time', 'loss'],
                 classes: List = [],
                 feature_importances: Optional[Callable] = None
                 ):
        """
        Initiate EpochTrainer.
        """
        self.logger = getLogger(self.__class__.__name__)
        self._max_num_epoches = max_num_epoches
        self._train_epoch = train_epoch
        self._predict = predict
        self._metrics = metrics
        self._save_model = save_model
        self._report_callback = report_callback
        self._report_hooks = report_hooks
        self._classes = classes
        self._feature_importances = feature_importances

    def train_step(self, train_data, epoch):
        if train_data is not None:
            train_data.shuffle(epoch)

        converge_flag = self._train_epoch(train_data)
        return converge_flag

    def supervised_metrics(self, preds, val_data):
        if self._metrics is not None and val_data and val_data.has_y() and preds is not None:
            if len(self._classes) <= 0:
                val_metrics = self._metrics(
                    preds.get_pred().to_numpy(),
                    val_data.get_y(first_only=True).to_numpy()
                )
            else:
                val_metrics = self._metrics(
                    preds.get_pred().to_numpy(),
                    val_data.get_y().to_numpy(),
                    classes=self._classes
                )
        else:
            val_metrics = None

        return val_metrics

    def unsupervised_metrics(self, preds, val_data):
        if self._metrics is not None:
            val_metrics = self._metrics(
                preds,
                val_data
            )
        else:
            val_metrics = None

        return val_metrics

    def evaluation_step(self, val_data, epoch, supervised: bool = True):
        preds = self._predict(val_data)

        if supervised:
            val_metrics = self.supervised_metrics(preds, val_data)
        else:
            val_metrics = self.unsupervised_metrics(preds, val_data)

        self.logger.info(
            f'Epoch {epoch} validation metrics are: {val_metrics}')
        return val_metrics

    def save_model(self, val_metrics, epoch):
        if self._save_model is not None:
            self.logger.info(f'Saving model parameters...')
            model_info = ModelInfo(
                model_id=None,
                model_attributes={
                    "metrics": val_metrics,
                    "verify_result": val_metrics,
                    "epoch_times": epoch,
                }
            )
            model_info = self._save_model(model_info)
            self.logger.info(f'Saved model info: {model_info}')
            return model_info
        else:
            return None

    def train(self,
              train_data: Optional[IBondDataFrame] = None,
              val_data: Optional[IBondDataFrame] = None,
              supervised: bool = True,
              epoch_offset: int = 0
              ) -> List[ModelInfo]:
        """
        Train data by epoches.
        TODO early stop

        Args:
            train_data, Optional[IBondDataFrame]
            val_data, Optional[IBondDataFrame]
            supervised, bool, whether used supervised metric to evaluate model.

        Returns:
            List[ModelInfo], a list ModelInfo of saved model.
        """
        model_infos = []
        set_hook = False
        if train_data is not None:
            if self._report_callback is not None:
                train_data.register_hooks(
                    [HOOKS[hook_name](self._report_callback) for hook_name in self._report_hooks]
                )
                set_hook = True
            train_data.set_context("max_num_epoch", self._max_num_epoches)

        for epoch in range(epoch_offset+1, self._max_num_epoches+1):
            self.logger.info(f'Start epoch {epoch}/{self._max_num_epoches}')
            if train_data is not None:
                train_data.set_context("current_epoch", epoch)

            converge_flag = self.train_step(train_data, epoch)
            val_metrics = self.evaluation_step(val_data, epoch, supervised)
            if val_metrics is not None:
                val_metrics['epoch'] = epoch - 1
                self._report_callback(
                    "validation_metrics",
                    json.dumps(val_metrics)
                )
            model_info = self.save_model(val_metrics, epoch)
            if model_info and self._feature_importances:
                feature_importances = self._feature_importances()
                model_info.update_attributes(
                    {
                        'feature_importances': feature_importances
                    }
                )

            if model_info is not None:
                model_infos.append(model_info)

            if converge_flag is True:
                self.logger.info(f'Converge epcoh is {epoch-1}')
                break

        if set_hook:
            # unregister hooks
            train_data.register_hooks([])

        return model_infos

    def unsupervised_train(self,
                           train_data: Optional[IBondDataFrame] = None,
                           val_data: Optional[IBondDataFrame] = None,
                           epoch_offset: int = 0
                           ) -> List[ModelInfo]:

        return self.train(train_data, val_data, False, epoch_offset)


class GuestGuidedEpochTrainer(EpochTrainer):
    def train(
        self,
        train_data: Optional[IBondDataFrame] = None,
        val_data: Optional[IBondDataFrame] = None,
        federal_info: Dict = {},
        supervised: bool = True,
        epoch_offset: int = 0
    ) -> List[ModelInfo]:
        model_infos = []
        set_hook = False

        if train_data is not None:
            if self._report_callback is not None:
                train_data.register_hooks(
                    [HOOKS[hook_name](self._report_callback) for hook_name in self._report_hooks]
                )
                set_hook = True
            train_data.set_context("max_num_epoch", self._max_num_epoches)

        # check federal info
        federal_info_parser = FederalInfoParser(federal_info)
        local_id = federal_info_parser.local_id
        # make channels
        continue_radio = Radio(
            station_id=federal_info_parser.major_guest,
            federal_info=federal_info_parser.federal_info,
            channels=['train_continue']
        )

        epoch = epoch_offset + 1
        while(True):
            self.logger.info(f'Start epoch {epoch}/{self._max_num_epoches}')

            # sync train_continue
            if local_id == federal_info_parser.major_guest:
                train_continue = (epoch <= self._max_num_epoches)
                continue_radio._train_continue_chan.broadcast(train_continue, tag='continue')
            else:
                train_continue = continue_radio._train_continue_chan.broadcast(tag='continue')

            if not train_continue:
                break

            if train_data is not None:
                train_data.set_context("current_epoch", epoch)

            converge_flag = self.train_step(train_data, epoch)
            val_metrics = self.evaluation_step(val_data, epoch, supervised)
            if val_metrics is not None:
                val_metrics['epoch'] = epoch - 1
                self._report_callback(
                    "validation_metrics",
                    json.dumps(val_metrics)
                )
            model_info = self.save_model(val_metrics, epoch)  # todo 这里后面修改一下吧，改成后续模型保存成功的report。
            if model_info and self._feature_importances:
                feature_importances = self._feature_importances()
                model_info.update_attributes(
                    {
                        'feature_importances': feature_importances
                    }
                )


            if model_info is not None:
                model_infos.append(model_info)

            if converge_flag is True:
                self.logger.info(f'Converge epcoh is {epoch - 1}')
                break

            epoch += 1

        if set_hook:
            # unregister hooks
            train_data.register_hooks([])

        return model_infos
