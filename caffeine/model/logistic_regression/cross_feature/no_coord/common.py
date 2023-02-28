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
from typing import Optional, Dict, List, Union
import math
import numpy as np
from abc import abstractmethod

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, root_validator
import tinygrad
from tinygrad.tensor import Tensor
from flex.constants import HE_OTP_LR_FT1, HE_LR_FP2

from caffeine.model.base_model import AbstractModel, JsonModel
from caffeine.model.base_model import ModelInfo
from caffeine.model.mixins import ModelSaver, FLEXUser
from caffeine.model.trainers import GuestGuidedEpochTrainer
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.utils.exceptions import DataMismatchError, ArgumentError, NotTrainedError
from caffeine.utils.federal_commu import FederalInfoParser, Radio
from caffeine.utils.metric import bcl_metrics


class HeteroLRParams(BaseModel):
    weights: List[float]
    with_bias: Optional[bool]
    federal_info: Dict


class SGDParams(BaseModel):
    learning_rate: PositiveFloat = 0.01
    batch_size: PositiveInt = 1024
    regularization: str = 'L2'
    alpha: PositiveFloat = 1.0
    max_epoch: PositiveInt = 10


class AdamParams(BaseModel):
    learning_rate: PositiveFloat = 0.01
    beta1: PositiveFloat = 0.9
    beta2: PositiveFloat = 0.99
    eps: PositiveFloat = 1.e-8
    batch_size: PositiveInt = 1024
    regularization: str = 'L2'
    alpha: PositiveFloat = 1.0
    max_epoch: PositiveInt = 10


class PaillierLikeParams(BaseModel):
    key_length: PositiveInt = 1024


class RandomNormalParams(BaseModel):
    mean: float = 0.0
    stddev: PositiveFloat = 1.0
    seed: Optional[int] = None


class OptimizerParams(BaseModel):
    type: str = 'Adam'
    parameters: Union[SGDParams, AdamParams] = SGDParams()

    @root_validator
    def check_type(cls, values):
        if values['type'] == 'Adam':
            values['parameters'] = AdamParams.parse_obj(values['parameters'])
        elif values['type'] == 'SGD':
            values['parameters'] = SGDParams.parse_obj(values['parameters'])
        return values

    class Config:
        smart_union = True


class EncryptorParams(BaseModel):
    type: str = 'Paillier'
    parameters: Union[PaillierLikeParams] = PaillierLikeParams()


class InitializerParams(BaseModel):
    type: str = 'random_normal'
    parameters: Union[RandomNormalParams] = RandomNormalParams()


class HeteroLRTrainParams(BaseModel):
    optimizer: OptimizerParams = OptimizerParams()
    encryptor: EncryptorParams = EncryptorParams()
    initializer: InitializerParams = InitializerParams()
    early_stop_param: Optional[dict] = None


class HeteroLogisticRegressionNoCoordMetaParams(BaseModel):
    train_param: Optional[HeteroLRTrainParams]
    federal_info: Optional[Dict]


class HeteroLogisticRegressionNoCoordBase(AbstractModel, ModelSaver, FLEXUser):
    _algo_info = {
        "algo_name": "HeteroLogisticRegression",
        "model_type": "algo_ml_2c_eval",
        "federate_type": 0  # 0 for federated model, 1 for local model
    }
    meta_param_model = HeteroLogisticRegressionNoCoordMetaParams
    meta_param_model_dict = {
        'train_param': HeteroLRTrainParams,
    }

    @ClassMethodAutoLog()
    def __init__(self,  meta_param: Dict, context: Context,  param: Optional[Dict] = None):
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
        # convert encryptor to security params
        self._meta_param['security_param'] = {}
        if 'train_param' in self._meta_param:
            encryptor_param = self._meta_param['train_param'].get(
                'encryptor',
                {}
            )
            if len(encryptor_param) > 0:
                if encryptor_param['type'] == 'Paillier':
                    key_length = encryptor_param.get('parameters', {}).get('key_length', 1024)
                    self._meta_param['security_param']['HE_OTP_LR_FT1'] = [
                        ["paillier", {"key_length": key_length}]
                    ]
                # if encryptor_param['type'] == 'CKKS':
                #     key_length = encryptor_param.get('parameters', {}).get('key_length', 8192)
                #     self._meta_param['security_param']['HE_OTP_LR_FT1'] = [
                #         ["ckks", {"key_length": key_length}]
                #     ]

            ### add early_stop config
            self._check_early_stop_param()
        self.logger.info(f'custom security param: {self._meta_param["security_param"]}')

        self._params = None
        self._feat_cols = None
        self._with_bias = None
        self._train_data_info = {}
        self._current_epoch = 0
        if 'federal_info' not in meta_param and param is None:
            raise ArgumentError(
                'Argument meta_param should contain key "federal_info" if param is None.')
        if param is not None:
            self.load_params(param)
        if 'train_param' not in meta_param and param:
            # predict mode
            # TODO
            self.init_protocols([
                HE_LR_FP2
            ])
        else:
            self.init_protocols([
                HE_OTP_LR_FT1,
                HE_LR_FP2
            ])

        self._federal_info_parser = FederalInfoParser(
            self._meta_param['federal_info'])

        self._radio = Radio(
            station_id=self._federal_info_parser.major_guest,
            federal_info=self._federal_info_parser.federal_info,
            channels=['minibatch_id', 'predict_batch_id', 'regularization', 'converge_flag']
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
              train_data: IBondDataFrame = None,
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
        
        num_epoches = self._meta_param['train_param']['optimizer']['parameters'].get('max_epoch', 10)
        trainer = GuestGuidedEpochTrainer(
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
            federal_info=self._federal_info_parser.federal_info,
            epoch_offset=self._current_epoch
        )

        self._after_train(train_data, val_data, model_infos)

        return model_infos
    
    @ClassMethodAutoLog()
    def _make_optimizer(self):
        optimizer_type = self._meta_param['train_param']['optimizer'].get('type', 'SGD')
        optimizer_params = self._meta_param['train_param']['optimizer']['parameters']
        self.logger.info('===============================')
        self.logger.info(f'Optimizer: {optimizer_type} with {optimizer_params}')
        self.logger.info('===============================')
        if optimizer_type == 'SGD':
            optimizer = tinygrad.optim.SGD(
                self._params['weights'],
                optimizer_params.get('learning_rate', 0.01)
            )
        elif optimizer_type == 'Adam':
            optimizer = tinygrad.optim.Adam(
                self._params['weights'],
                lr = optimizer_params.get('learning_rate', 0.01),
                b1 = optimizer_params.get('beta1', 0.9),
                b2 = optimizer_params.get('beta2', 0.99),
                eps = optimizer_params.get('eps', 1.e-8)
            )
        else:
            raise Exception(f'Unkown optimizer type: {optimizer_type}')
        return optimizer

    @ClassMethodAutoLog()
    def _init_partner_params(
        self, 
        feat_cols: Optional[List[str]], 
        bias: int=0,
        initializer_params: Optional[Dict] = None
    ):
        """
        Init or check model parameters.

        Args:
            feat_cols: optional List[str], a list of feature column names.
        """
        if self._params is None or feat_cols != self._feat_cols:
            self.logger.info(
                f'Initialize model parameters by feature columns {feat_cols}.')
            self.logger.info(
                f'Initializer_params: {initializer_params}')
            if initializer_params:
                initializer_type = initializer_params.get('type', 'random_normal')
                initializer_parameters = initializer_params.get('parameters', {})
                if initializer_type == 'random_normal':
                    self._params = {
                        'weights': [Tensor(np.random.normal(
                            loc=initializer_parameters.get('mean', 0.),
                            scale=initializer_parameters.get('stddev', 1.),
                            size=len(feat_cols)+bias
                        ).flatten())], #bias
                        'feat_cols': feat_cols
                    }
                elif initializer_type == 'zeros':
                    self._params = {
                        'weights': [Tensor(np.zeros(len(feat_cols)+bias).flatten())], #bias
                        'feat_cols': feat_cols
                    }
                elif initializer_type == 'random_uniform':
                    self._params = {
                        'weights': [Tensor(
                            np.random.uniform(
                                low = initializer_parameters.get('low', 0.),
                                high = initializer_parameters.get('high', 1.),
                                size = len(feat_cols)+bias
                            ).flatten()
                        )], #bias
                        'feat_cols': feat_cols
                    }
                else:
                    raise Exception(f'Unkown initializer type: {initializer_type}.')
            else:
                self._params = {
                    'weights': [Tensor(np.zeros(len(feat_cols)+bias).flatten())], #bias
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
