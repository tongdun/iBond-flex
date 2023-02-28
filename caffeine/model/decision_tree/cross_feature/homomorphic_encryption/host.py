#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name:                                                                           
#                                                                                              
#  Create date:  2020/12/29
from typing import Dict, Optional

from caffeine.model.base_model import ModelInfo
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from .common import HeteroDTBase
from caffeine.utils.exceptions import NotTrainedError, LossTypeError
from caffeine.utils.loss import *


class HeteroDTHost(HeteroDTBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_param: Dict, context: Context, param: Optional[Dict] = None):
        """
        Initiate Model.

        Args:
            meta_param, a dict of meta parameters:
            {

            }
        """
        super().__init__(meta_param, context, param)

    @ClassMethodAutoLog()
    def train(self, train_data: IBondDataFrame, val_data: Optional[IBondDataFrame] = None, model_idx: int = 0) -> ModelInfo:
        '''
        Train decision tree at host.

        Args:
            data: IBondDataFrame, shape[n, m]. Train data.
            model_idx: int. Model index tag for identify model while ensembling models.

        Returns:
            ModelInfo

        '''

        self._before_train(train_data)

        self._before_one_tree(train_data)

        self.train_dt(train_data)

        model_infos = self._after_train(val_data)

        return model_infos

    @ClassMethodAutoLog()
    def predict(self, data: Optional[IBondDataFrame] = None, predict_id: Optional[str] = '*'):
        """
        Model prediction interface.

        Args:
            data: ibond dataframe.

        Return:
            prediction: ibond dataframe, output predictions,  If the guest data has id col, return id and predictions, else return
            predictions only.
        """
        #check guest/host predict data shape is equal
        data_shape = data.shape[0] if data is not None else 0
        # total_val_data = self._protocols[OTP_PN_FL].param_negotiate(param='equal', data=data_shape, tag=predict_id + 'he_dt_pred_data_shape')
        if data_shape == 0:
            self.logger.info(f'host predict data is none')
            return

        self.logger.info(f'predict current trees:{self.nodes}')
        # check training status
        if self.nodes is None:
            raise NotTrainedError('Host cross feature decision tree model has not been trained.')

        pred = self.predict_dt(data, predict_tree_id=predict_id)

        prediction = self._radio._hetero_he_dt_predictions_chan.broadcast(tag=predict_id)

        return prediction
