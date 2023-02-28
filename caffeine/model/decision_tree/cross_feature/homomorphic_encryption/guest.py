#!/usr/bin/python3
#
#  Copyright 2021 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name:                                                                           
#                                                                                              
#  Create date:  2021/1/4
from typing import Dict, Optional

import pandas as pd

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from .common import HeteroDTBase
from caffeine.utils.exceptions import NotTrainedError, LossTypeError
from caffeine.utils.loss import *


class HeteroDTGuest(HeteroDTBase):
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
    def train(self, train_data: IBondDataFrame, val_data: Optional[IBondDataFrame] = None):
        '''
        Train decision tree at guest.

        Args:
            data: IBondDataFrame, shape[n, m]. Train data.
            predictions: IBondDataFrame, shape[n, 1]. Predictions.
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
        '''
        Predict decision tree at guest.

        Args:
            data: IBondDataFrame, shape[n, m]. Predict data.
            predictions: IBondDataFrame, shape[n, 1]. Predictions.
            model_idx: int, Model index tag for identify model while ensembling models.
            with_sigmoid: bool, if value is True, apply sigmoid to predictions.

        Returns:
            ModelInfo

        '''
        # check guest/host predict data is equal or not
        data_shape = data.shape[0] if data is not None else 0
        # total_val_data = self._protocols[OTP_PN_FL].param_negotiate(param='equal', data=data_shape, tag=predict_id + 'he_dt_pred_data_shape')
        if data_shape == 0:
            self.logger.info(f'guest predict data is none')
            return

        self.logger.info(f'predict current trees:{self.nodes}')
        # check training status
        if self.nodes is None:
            raise NotTrainedError('Guest cross feature decision tree model has not been trained.')

        preds = self.predict_dt(data, predict_tree_id=predict_id)

        # init loss
        if self.loss_type == 'BCELoss':
            self.loss = BCELoss()
        elif self.loss_type == 'MSELoss':
            self.loss = MSELoss()
        elif self.loss_type == 'HingeLoss':
            self.loss = HingeLoss()
        else:
            raise LossTypeError("loss type not support yet")

        preds = self.loss.predict(preds)

        if data.has_id():
            output = data.get_id().to_pandas()
            output['preds'] = preds
        else:
            output = pd.DataFrame({'preds': preds})

        self.logger.info(f'Predictions of cross feature DT are: {output}')

        prediction = self._context.create_dataframe(
            pdf=output,
            data_desc={
                'id_desc': data.data_desc['id_desc'],
                'pred': ['preds'],
            }
        )

        self._radio._hetero_he_dt_predictions_chan.broadcast(
            prediction,
            tag=predict_id
        )

        return prediction

