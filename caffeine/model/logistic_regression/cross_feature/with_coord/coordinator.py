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
#                                                                                              
#  File name: coordinator.py                                                                         
#                                                                                              
#  Create date: 2020/11/24                                                                               
#
from flex.constants import HE_OTP_LR_FT2, HE_LR_FP, OTP_SA_FT, OTP_PN_FL
from typing import Optional, Dict, Any

from tinygrad.tensor import Tensor

from caffeine.utils import ClassMethodAutoLog
from caffeine.utils import IBondDataFrame, Context
from caffeine.utils.exceptions import DataCheckError
from caffeine.utils.dataframe import parse_ibonddf
from .common import HeteroLogisticRegressionBase


class HeteroLogisticRegressionCoord(HeteroLogisticRegressionBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_param: Dict, context: Context, param: Optional[Dict] = None):
        """
        Initiate Model.

        Args:
            meta_param: a dict of meta parameters:
            {
                'train_param': {
                    'num_epoches': int, optional, number of epoches.
                },
                OR
                'predict_param': {
                },
                'sec_param': securtiy,
                'federal_info': dict, federal info
            }
            context: Context.
            param: optional dict, if not None, load model from this dict.
        """
        super().__init__(meta_param, context, param)
        self.epoch = 1 # TODO

    @ClassMethodAutoLog()
    def _init_params(self):
        """
        Init or check model parameters.

        Args:
            feat_cols: optional List[str], a list of feature column names.
        """
        self._protocols[OTP_SA_FT].exchange()
        self.logger.info(f'Initialize an empty model.')
        self._params = {
            'weights': [Tensor([])],
            'feat_cols': []
        }
        self._feat_cols = []
        return

    @ClassMethodAutoLog()
    def predict(self, data: Any = None, predict_id: Optional[str] = '*') -> IBondDataFrame:
        """
        Model predicting interface. Output predictions for data.

        Args:
            data: any type, will be ignored.
            predict_id: predict Id
        """
        self.logger.info(f'Start receive num predict batches...')
        predict_num_batches = self._protocols[OTP_PN_FL].param_negotiate(
            param='equal',
            data=None,
            tag=predict_id
        )
        self.logger.info(f'Total number of predict batches: {predict_num_batches}')
        if predict_num_batches <= 0 or predict_num_batches is None:
            return

        # loop
        for i in range(predict_num_batches):
            self.logger.info(f'>>Start batch {i + 1}/{predict_num_batches}')
            self._protocols[HE_LR_FP].exchange(tag=predict_id)

        # NOTE broadcast predictions
        prediction = self._radio._hetero_lr_predictions_chan.broadcast(
            tag=predict_id
        )

        return prediction

    @ClassMethodAutoLog()
    def train_epoch(self, data=None):
        """
        Model training interface. Train an epoch of data, update self model
        parameters.

        Args:
            data: must be None.
        """
        self._init_params()

        train_num_batches = self.negotiate_num_batches()
        self.logger.info(f'Total number of train batches: {train_num_batches}')

        # loop
        iter_num = 0
        for _ in data.dummy_batches(train_num_batches):
            self._protocols[HE_OTP_LR_FT2].exchange(
                epoch=self.epoch,
                iter=iter_num+1
            )
            # New
            remote_loss = self._radio._hetero_lr_training_loss_chan.broadcast()
            data.set_context("loss", remote_loss)

            # New: add convergence judge
            self.converge_flag = self._radio._hetero_lr_converge_flag_chan.broadcast()
            if self.converge_flag is True:
                self.logger.info(f'**************************coord converge_flag is True')
                break

            iter_num += 1

        self.converge_flag = self._radio._hetero_lr_converge_flag_chan.broadcast()

        self.epoch += 1

        return self.converge_flag
