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
#  File name: train
#
#  Create date: 2020/11/24
#
import math
import numpy as np
import tinygrad
from abc import abstractmethod
from flex.constants import HE_OTP_LR_FT2, HE_LR_FP, OTP_SA_FT, OTP_PN_FL
from flex.utils.check_security_params import validator, valid_he_otp_lr_ft2, valid_he_lr_fp
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from tinygrad.tensor import Tensor
from typing import Optional, Dict, List, Union

from caffeine.model.base_model import AbstractModel, JsonModel
from caffeine.model.base_model import ModelInfo
from caffeine.model.mixins import ModelSaver, FLEXUser
from caffeine.model.trainers import EpochTrainer
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.utils.exceptions import DataMismatchError, ArgumentError, NotTrainedError
from caffeine.utils.federal_commu import FederalInfoParser, Radio
from caffeine.utils.metric import bcl_metrics


class HeteroLRParams(BaseModel):
    weights: List[float]
    with_bias: Optional[bool]
    federal_info: Dict


class HeteroLRTrainParams(BaseModel):
    learning_rate: PositiveFloat = 0.01
    batch_size: PositiveInt = 1024
    num_epoches: PositiveInt = 10
    with_bias: Optional[bool] = None
    early_stop_param: Optional[dict]=None

    class Config:
        schema_extra = {
            'expose': ['learning_rate', 'batch_size', 'num_epoches']
        }


class HeteroLRPredictParams(BaseModel):
    batch_size: PositiveInt = 4096

    class Config:
        schema_extra = {
            'expose': ['batch_size']
        }


class HeteroLRSecurityParams(BaseModel):
    HE_OTP_LR_FT2: Optional[List] = [["paillier", {"key_length": 1024}]]
    HE_LR_FP: Optional[List] = [["paillier", {"key_length": 1024}]]

    _he_otp_lr_ft2 = validator('HE_OTP_LR_FT2', allow_reuse=True)(valid_he_otp_lr_ft2)
    _he_lr_fp = validator('HE_LR_FP', allow_reuse=True)(valid_he_lr_fp)

    class Config:
        schema_extra = {
            'expose': [eval('HE_OTP_LR_FT2'), eval('HE_LR_FP')]
        }


class HeteroLogisticRegressionMetaParams(BaseModel):
    train_param: Optional[HeteroLRTrainParams]
    predict_param: Optional[HeteroLRPredictParams]
    security_param: Optional[HeteroLRSecurityParams]
    federal_info: Optional[Dict]


class HeteroLogisticRegressionBase(AbstractModel, ModelSaver, FLEXUser):
    _algo_info = {
        "algo_name": "HeteroLogisticRegression",
        "model_type": "algo_ml_2c_eval",
        "federate_type": 0  # 0 for federated model, 1 for local model
    }
    meta_param_model = HeteroLogisticRegressionMetaParams
    meta_param_model_dict = {
        'train_param': HeteroLRTrainParams,
        'predict_param': HeteroLRPredictParams,
        'security_param': HeteroLRSecurityParams
    }

    @ClassMethodAutoLog()
    def __init__(self, meta_param: Dict, context: Context, param: Optional[Dict] = None):
        """
        Common init operations for all participants.

        Args:
            meta_param: dict, meta parameters for this model.
            context: Context, context, e.g. wafer session.
            param: optional dict, if not None, load model from this dict.
        """
        super().__init__(meta_param, context=context)

        # check meta_param
        self._parse_meta_param()

        self._params = None
        self._feat_cols = None
        self._with_bias = None
        self._current_epoch = 0
        self._train_data_info = {}
        if 'federal_info' not in meta_param and param is None:
            raise ArgumentError(
                'Argument meta_param should contain key "federal_info" if param is None.')
        if param is not None:
            self.load_params(param)

        if 'train_param' in self._meta_param:
            ### add early_stop config
            self._check_early_stop_param()
            self.logger.info(f'**************************Init early_stop param:{self.early_stop_param}')

        self.init_protocols([
            OTP_SA_FT,
            HE_OTP_LR_FT2,
            HE_LR_FP,
            OTP_PN_FL
        ])

        self._federal_info_parser = FederalInfoParser(
            self._meta_param['federal_info'])

        self._radio = Radio(
            station_id=self._federal_info_parser.major_guest,
            federal_info=self._federal_info_parser.federal_info,
            channels=['hetero_lr_predictions', 'hetero_lr_training_loss', 'hetero_lr_converge_flag']
        )

    @property
    def params(self):
        return self._params

    @property
    def weights(self):
        if self._params and 'weights' in self._params:
            return self._params['weights'][0].data.flatten()
        else:
            return None

    @ClassMethodAutoLog()
    def feature_importances(self):
        if self.weights is not None and self._feat_cols is not None and self._with_bias is not None:
            weights_square = self.weights ** 2 / (np.sum(self.weights ** 2) + 1.e-10)
            if self._with_bias:
                return dict(zip(['bias']+self._feat_cols, weights_square))
            else:
                return dict(zip(self._feat_cols, weights_square))
        else:
            return {}

    def train_epoch(self, *args, **kwargs):
        raise NotImplementedError

    @ClassMethodAutoLog()
    def train(self,
              train_data: Optional[IBondDataFrame] = None,
              val_data: Optional[IBondDataFrame] = None
              ) -> List:
        """
        Train cross-feature logistic regression.

        Args:
            train_data: optional ibond dataframe, train data.
            val_data: optional ibond dataframe, validation data.

        Returns:
            List: a list of saved model infos.
        """
        self._before_train(train_data, val_data)

        num_epoches = self._meta_param['train_param']['num_epoches']
        trainer = EpochTrainer(
            max_num_epoches=num_epoches,
            train_epoch=self.train_epoch,
            save_model=self._save_params,
            predict=self.predict,
            metrics=bcl_metrics,
            report_callback=self._context.report,
            feature_importances=self.feature_importances
        )
        if train_data is None:
            train_data = self._context.create_dataframe(None)

        model_infos = trainer.train(
            train_data, 
            val_data,
            epoch_offset=self._current_epoch
        )

        self._after_train(train_data, val_data, model_infos)

        return model_infos

    @ClassMethodAutoLog()
    def train_epoch(self, data: Optional[IBondDataFrame] = None):
        raise NotImplementedError("Method train_epoch is not implemented.")

    @ClassMethodAutoLog()
    def _make_optimizer(self):
        optimizer = tinygrad.optim.SGD(
            self._params['weights'],
            self._meta_param['train_param']['learning_rate']
        )
        return optimizer

    @ClassMethodAutoLog()
    def negotiate_num_batches(self, data_num: int = None, batch_size: int = None) -> int:
        """
        Sync train num batches
        Args:
            data_num, int, the total number of training data. for example len(data).
            batch_size, int, a small batch (n>0) 
        Returns:
            int, number of batches for training(n>0)
        """
        if data_num is None and batch_size is None:
            train_num_batches = self._protocols[OTP_PN_FL].param_negotiate(
                param='equal',
                data=None
            )
        else:
            if not isinstance(data_num, int):
                raise ValueError(
                    f"data_num({type(data_num)}) must be integer.")

            if not isinstance(batch_size, int):
                raise ValueError(
                    f"batch_size({type(batch_size)}) must be integer.")

            if batch_size <= 0:
                raise ValueError(
                    f"batch_size must be larger than 1. current batch_size is {batch_size}.")

            local_num_batches = int(math.ceil(data_num / float(batch_size)))

            train_num_batches = self._protocols[OTP_PN_FL].param_negotiate(
                param='equal',
                data=local_num_batches
            )

        self.logger.info(f'Toal number of train batches: {train_num_batches}')
        return train_num_batches

    @ClassMethodAutoLog()
    def _init_partner_params(self, feat_cols: Optional[List[str]], bias: int = 0):
        """
        Init or check model parameters.

        Args:
            feat_cols: optional List[str], a list of feature column names.
        """
        self._sum_feats = int(self._protocols[OTP_SA_FT].exchange(
            np.array([len(feat_cols)])
        )[0] * 2)
        if self._params is None or feat_cols != self._feat_cols:
            self.logger.info(
                f'Initialize model parameters by feature columns {feat_cols}.')
            self._params = {
                'weights': [Tensor(np.zeros(len(feat_cols) + bias).flatten())],  # bias
                'feat_cols': feat_cols
            }
            self._feat_cols = feat_cols
        else:
            self.logger.info(f'Pass model initialization.')

    @ClassMethodAutoLog()
    def _save_params(self,
                     model_info: Optional[ModelInfo] = None
                     ) -> ModelInfo:
        """
        Save parameters to middleware.
        """
        if self._params is None:
            return
        lr_mod, _ = self.make_module(
            params=HeteroLRParams(
                weights=self.weights.tolist(),
                with_bias=self._with_bias,
                federal_info=self._meta_param['federal_info']
            )
        )
        if model_info is None:
            model_info = ModelInfo(
                None, {}
            )
        model_info.update_attributes(self._algo_info)
        model_info.update_attributes(self._train_data_info)

        model_id = self.save_modules(
            [lr_mod],
            self._meta_param['federal_info'],
            model_info
        )
        model_info.update_id(model_id)

        return model_info

    @ClassMethodAutoLog()
    def load_params(self, params: Dict):
        """
        Load parameters into this model.

        Args:
            params: the dict representation of the model.

        -----

        **Example:**

        >>> params = {
                'model_type': 'HeteroLogisticRegressionGuest',
                'versions': {
                    'caffeine': '0.1',
                    'flex': '1.1',
                    'wafer': '1.0'
                },
                'feat_cols': [Age, Sex],
                'modules': [
                    {
                        'module_id': 'Module-HeteroLogisticRegressionGuest-2020-01-01-10-12-33-12345',
                        'module_type': 'Module-HeteroLogisticRegressionGuest',
                        'params': {
                            'weights': [0.223, 0.1],
                        }
                    }
                ]
            }
        >>> model = HeteroLogisticRegressionGuest(meta_param)
        >>> model.load_params(params)
        """
        self.logger.info(f'Parsing params: {params}...')
        model = JsonModel.parse_obj(params)

        weights = np.array(model.modules[0].params['weights'])
        self._with_bias = model.modules[0].params['with_bias']
        self.logger.info(f'Loading weights: {weights}...')
        self._params = {
            'weights': [Tensor(weights.flatten())]
        }

        feat_cols = model.feat_cols
        self.logger.info(f'Loading feat_cols: {feat_cols}...')
        self._feat_cols = feat_cols

        trained_epoches = model.attributes.get('epoch_times', 0) # TODO 0 or 1
        self._current_epoch = trained_epoches

        if 'federal_info' not in self._meta_param:
            federal_info = model.modules[0].params['federal_info']
            self.logger.info(
                f'Loading federal_info from model: {federal_info}...')
            self._meta_param['federal_info'] = federal_info
            self.logger.info(
                f'Use loaded federal_info: {self._meta_param["federal_info"]}')
        else:
            self.logger.info(
                f'Use custom federal_info: {self._meta_param["federal_info"]}')

        self.logger.info(f'Model loaded.')

    def _check_train_status(self):
        """
        check training status
        Returns:

        """
        if self._params is None or self._feat_cols is None:
            raise NotTrainedError('This model has not been trained.')

    def _sync_predict_batch(self, data: IBondDataFrame, predict_id: Optional[str] = '*') -> int:
        """
        sync predict num batches
        Args:
            data: ibond dataframe, input data for prediction, has batch_size
                rows, should contain features for training.
            predict_id: the id of predict

        Returns: the batches of predict

        """
        # get predict num rows
        predict_num_rows = 0 if not data else data.shape[0]
        # NOTE cannot return here if empty, must inform coordinator first.
        predict_num_batches = self._protocols[OTP_PN_FL].param_negotiate(
            param='equal',
            data=int(
                math.ceil(predict_num_rows / float(self._predict_batch_size))
            ),
            tag=predict_id
        )
        self.logger.info(
            f'Total number of batches in prediction: {predict_num_batches}')
        return predict_num_batches
