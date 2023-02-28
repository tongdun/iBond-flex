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
#  File name: host.py
#
#  Create date: 2020/11/24
#
import math
import numpy as np
import pandas as pd
from flex.constants import HE_OTP_LR_FT1, HE_LR_FP2
from logging import getLogger
from tinygrad.tensor import Tensor
from typing import Optional, List, Dict, Union

from caffeine.utils import ClassMethodAutoLog
from caffeine.utils import IBondDataFrame, Context
from caffeine.utils.exceptions import NotTrainedError
from caffeine.utils.loss import BCELoss
from .common import HeteroLogisticRegressionNoCoordBase


class HeteroLogisticRegressionNoCoordHost(HeteroLogisticRegressionNoCoordBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_param: Dict, context: Context, param: Optional[Dict] = None):
        """
        Initiate Model.

        Args:
            meta_param: a dict of meta parameters:
            {
                'train_param': {
                    'learning_rate': float, init learning rate
                    'batch_size': int, optimizer batch size
                },
                OR
                'predict_param': {
                    'batch_size': int, prediction batch size
                },
                'security': security,
                'federal_info': dict, federal info
            }
            context: Context
            param: optional dict, if not None, load model from this dict.
        """
        super().__init__(meta_param, context, param)
        if 'train_param' in self._meta_param:
            self._regularization_type = self._meta_param['train_param']['optimizer']['parameters'].get('regularization', None)
            self._alpha = self._meta_param['train_param']['optimizer']['parameters'].get('alpha', 1.)
            self._initializer_params = self._meta_param['train_param'].get('initialzer', {})
            self._with_bias = False #self._meta_param['train_param'].get('with_bias', False)

    @ClassMethodAutoLog()
    def _init_params(self, feat_cols: Optional[List[str]], bias: int=0, initializer_params: Optional[Dict]=None):
        return self._init_partner_params(feat_cols, bias=bias, initializer_params=initializer_params)

    @ClassMethodAutoLog()
    def predict(self, data: Optional[IBondDataFrame],
                predict_id: Optional[str] = '*') -> IBondDataFrame:
        """
        Model predicting interface. Output predictions for data.

        Args:
            data: ibond dataframe, input data for prediction, has batch_size
                rows, should contain features for training.
            predict_id: the id of predict
        """
        # check training status
        self._check_train_status()
        # if data is empty, return
        # TODO D data
        if data is None or data.shape[0] <= 0:
            return 

        id_cols = data.data_desc['id_desc']
        if True: # reserve for loop
            self.logger.info(f'>>Start batch')
            batch_id = self._radio._predict_batch_id_chan.broadcast()
            if len(batch_id) == 0 or len(batch_id[0]) == 0:
                self.logger.warning('Received empty batch ids, directly return.')
                return None
            batch_id_ibond = self._context.create_dataframe(
                pd.DataFrame(
                    batch_id,
                    columns = id_cols
                )
            )
            # batch via ibond dataframe join
            batch = data.join(
                other = batch_id_ibond,
                key = id_cols,
                how = 'right'
            )
            feature = batch[self._feat_cols].to_numpy()
            if self._with_bias:
                feature = np.concatenate(
                    (np.ones((feature.shape[0], 1)), feature),
                    axis=1
                )
            logits = np.dot(feature, self.weights).flatten()
            self._protocols[HE_LR_FP2].exchange(
                logits,
                tag=predict_id
            )

        return

    @ClassMethodAutoLog()
    def train_epoch(self, data: IBondDataFrame):
        """
        Model training interface. Train an epoch of data, update self model
        parameters.

        Args:
            data: ibond dataframe, input training data.
        """
        # init and check params
        feat_cols = self._train_data_info['feat_cols']
        id_cols = self._train_data_info['id_cols']
        self._init_params(
            feat_cols,
            bias=1 if self._with_bias else 0,
            initializer_params=self._initializer_params
        )
        # update self._train_data_info
        self._train_data_info.update({
            "local_num_id": len(id_cols),
            "local_num_x": len(feat_cols),
            "local_num_y": 0,
        })

        optimizer = self._make_optimizer()
        epoch = data._context["current_epoch"]
        iter = 1
        # NOTE guest guided batch generation
        while(True):
            batch_id = self._radio._minibatch_id_chan.broadcast()
            if len(batch_id) <= 0:
                break
            if self._regularization_type == 'L2':
                regularization = float(np.sum(self.weights ** 2) * 0.5)
            else:
                regularization = 0.
            self._radio._regularization_chan.gather(regularization, tag='reg')

            batch_id_ibond = self._context.create_dataframe(
                pd.DataFrame(
                    batch_id,
                    columns = id_cols
                )
            )
            # batch via ibond dataframe join
            batch = data.join(
                other = batch_id_ibond,
                key = id_cols,
                how = 'right'
            )
            feature = batch[self.feat_cols].to_numpy()
            if self._with_bias:
                feature = np.concatenate(
                    (np.ones((feature.shape[0], 1)), feature),
                    axis=1
                )
            gradient = self._protocols[HE_OTP_LR_FT1].exchange(
                self.weights,
                feature,
                regularization_type = self._regularization_type,
                alpha = self._alpha,
                epoch = epoch,
                iter = iter
            )
            optimizer.zero_grad()
            self._params['weights'][0].grad = gradient
            optimizer.step()
            iter += 1
            self.logger.info(f"Current weights are: {self.weights}.")

            self.converge_flag = self._radio._converge_flag_chan.broadcast()
            if self.converge_flag is True:
                self.logger.info(f'**************************host converge_flag is True')
                break

        self.converge_flag = self._radio._converge_flag_chan.broadcast()

        return self.converge_flag






