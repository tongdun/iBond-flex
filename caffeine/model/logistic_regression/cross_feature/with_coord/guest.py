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
#  File name: guest.py
#
#  Create date: 2020/11/24
#
import math
import numpy as np
import pandas as pd
from flex.constants import HE_OTP_LR_FT2, HE_LR_FP, OTP_SA_FT, OTP_PN_FL
from tinygrad.tensor import Tensor
from typing import Optional, List, Dict, Union

from caffeine.utils import ClassMethodAutoLog
from caffeine.utils import IBondDataFrame, Context
from caffeine.utils.activations import sigmoid
from caffeine.utils.exceptions import NotTrainedError
from caffeine.utils.loss import BCELoss
from .common import HeteroLogisticRegressionBase


class HeteroLogisticRegressionGuest(HeteroLogisticRegressionBase):
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
                    'num_epoches': int, optional, number of epoches
                },
                OR
                'predict_param': {
                    'batch_size': int, prediction batch size },
                'security': security
                'federal_info': dict, federal info
            }
            context: Context
            param: optional dict, if not None, load model from this dict.
        """
        super().__init__(meta_param, context, param)
        if 'train_param' in self._meta_param:
            self._train_batch_size = self._meta_param['train_param']['batch_size']
            self._predict_batch_size = self._meta_param['train_param']['batch_size']
            self._with_bias = self._meta_param['train_param'].get('with_bias', True)
        if 'predict_param' in self._meta_param:
            self._predict_batch_size = self._meta_param['predict_param']['batch_size']

    @ClassMethodAutoLog()
    def _init_params(self, feat_cols: Optional[List[str]], bias: int=1):
        return self._init_partner_params(feat_cols, bias=bias)

    @ClassMethodAutoLog()
    def predict(self, data: IBondDataFrame,
                predict_id: Optional[str] = '*') -> Optional[IBondDataFrame]:
        """
        Model predicting interface. Output predictions for data.

        Args:
            data: ibond dataframe, input data for prediction, has batch_size
                rows, should contain features for training.
            predict_id: the id of predict

        Returns:
            ibond dataframe: output predictions, has batch_size rows. If the
                training data has id col, return id and predictions, else return
                predictions only. Predictions are probabilities for postive.
        """
        # check training status
        self._check_train_status()
        # sync validation num batches
        predict_num_batches = self._sync_predict_batch(data, predict_id)
        # if data is empty, return
        if predict_num_batches <= 0 or predict_num_batches is None:
            return
        # loop
        output = None
        for i, batch in enumerate(data.batches(self._predict_batch_size)):
            self.logger.info(f'>>Start batch {i + 1}/{predict_num_batches}')
            feature = batch[self._feat_cols].to_numpy()
            if self._with_bias:
                feature = np.concatenate(
                    (np.ones((feature.shape[0], 1)), feature),
                    axis=1
                )
            logits = np.dot(feature, self.weights).flatten()
            batch_local_preds = sigmoid(logits)
            fed_logits = self._protocols[HE_LR_FP].exchange(
                logits,
                tag=predict_id
            )
            batch_preds = sigmoid(fed_logits)

            if batch.has_id():
                batch_output = batch.get_id().to_pandas()
                batch_output['preds'] = batch_preds.flatten()
                batch_output['local_preds'] = batch_local_preds.flatten()
            else:
                batch_output = pd.DataFrame(
                    {
                        'preds': batch_preds.flatten(),
                        'local_preds': batch_local_preds.flatten()
                    }
                )

            if i == 0:
                output = batch_output
            else:
                output = pd.concat([output, batch_output])

        self.logger.info(f'Predictions are: {output}')
        prediction = self._context.create_dataframe(
            pdf=output,
            data_desc={
                # TODO, data_desc needs a class to standardize initial operation.
                'id_desc': data.data_desc['id_desc'],
                'pred': ['preds'],
                'local_pred': ['local_preds']
            }
        )
        # NOTE broadcast predictions
        self._radio._hetero_lr_predictions_chan.broadcast(
            prediction,
            tag=predict_id
        )

        return prediction

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
            bias=1 if self._with_bias else 0
        )
        # update self._train_data_info
        self._train_data_info.update({
            "local_num_id": len(id_cols),
            "local_num_x": len(feat_cols),
            "local_num_y": 1,
            "fed_num_id": len(id_cols),
            "fed_num_x": self._sum_feats,
            "fed_num_y": 1
        })

        train_num_batches = self.negotiate_num_batches(
            data.shape[0], self._train_batch_size)

        # loop
        optimizer = self._make_optimizer()
        bce_loss = BCELoss()
        total_loss = 0.0
        iter_num = 0

        epoch = data._context["current_epoch"]

        for batch in data.batches(self._train_batch_size, max_num=train_num_batches):
            feature = batch[self.feat_cols].to_numpy()
            if self._with_bias:
                feature = np.concatenate(
                    (np.ones((feature.shape[0], 1)), feature),
                    axis=1
                )
            label = batch.get_y(first_only=True).to_numpy().flatten()
            predict, gradient = self._protocols[HE_OTP_LR_FT2].exchange(
                self.weights,
                feature,
                label,
                epoch=epoch,
                iter=iter_num+1
            )
            optimizer.zero_grad()
            self._params['weights'][0].grad = gradient
            optimizer.step()
            self.logger.info(f"Current weights are: {self.weights}.")
            local_loss = bce_loss.loss(predict, label)
            self.logger.info(f'Current loss is: {local_loss}')

            # New
            self._radio._hetero_lr_training_loss_chan.broadcast(local_loss)
            data.set_context("loss", local_loss)

            total_loss += local_loss
            # New: add convergence judge
            iter_num += 1
            if self.early_stop_param['early_stop'] is True and self.early_stop_param['early_stop_step'] == 'iter':
                self.converge_flag = self.converge_func.is_converge(local_loss)
            self._radio._hetero_lr_converge_flag_chan.broadcast(self.converge_flag)

            self.logger.info(f'**************************guest batch converge_flag:{self.converge_flag}')

            if self.converge_flag is True:
                self.logger.info(f'**************************guest converge_flag is True')
                break

        epoch_avg_loss = total_loss / float(iter_num)
        self.logger.info(f'epoch {epoch} avg loss:{epoch_avg_loss}')

        if self.early_stop_param['early_stop'] is True and self.early_stop_param['early_stop_step'] == 'epoch':
            self.converge_flag = self.converge_func.is_converge(epoch_avg_loss)

        self._radio._hetero_lr_converge_flag_chan.broadcast(self.converge_flag)

        return self.converge_flag
