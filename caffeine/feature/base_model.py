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
#                                                                                              #                                                                                              
#  Create date: 2021/12/24                                                                               
#
from abc import ABCMeta, abstractmethod
from logging import getLogger
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Any
from re import finditer

from caffeine.utils import ClassMethodAutoLog
from caffeine.utils import Context
from caffeine.utils import IBondDataFrame
from caffeine.utils import config
from caffeine.utils.dataframe import parse_ibonddf


class MetaParams(BaseModel):
    train_param: Any
    predict_param: Any
    security_param: Any
    federal_info: Any


class FeatureAbstractModel(metaclass=ABCMeta):
    meta_param_model = MetaParams
    meta_param_model_dict = {
        'train_param': BaseModel,
        'predict_param': BaseModel
    }

    @abstractmethod
    def __init__(self, meta_params: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        Initiate Model.

        Args:
            meta_param: a dict of meta parameters.
            context: Context, context of the model to save models etc..
                e.g. a wafer session.
        """
        self.logger = getLogger(self.__class__.__name__)
        self._meta_params = meta_params
        self._context = context

        self._feat_cols = None

    def _parse_meta_params(self):
        if self.meta_param_model is not None:
            try:
                self._meta_params = self.meta_param_model.parse_obj(
                    self._meta_params
                ).dict(exclude_none=False)
                self.logger.info(f"meta_params {self._meta_params}")
            except:
                self.logger.error(f'!!!Parse meta_param {self._meta_params} failed!!!')
                raise Exception('Parse meta parameters failed.')
        else:
            self.logger.info(f'Empty meta_param model, skip checking.')


    @classmethod
    def class_algo_name(cls):
        return ''.join([m.group(0) for m in finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', cls.__name__)][:-1])

    @property
    def algo_name(self):
        return self.class_algo_name()

    @property
    def feat_cols(self):
        return self._feat_cols

    # @abstractmethod
    def train(self, train_data: Optional[IBondDataFrame], 
                verify_data: Optional[IBondDataFrame]=None, 
                feat_info: Optional[Dict]=dict()):
        """
        Model training interface. Train via data, update self model
        parameters.

        Args:
            train_data: optional ibond dataframe, input training data.
            verify_data: optional ibond dataframe, input validation data.
            feat_infos: dict, feature infos from previous operators.
        """
        pass

    # @abstractmethod
    def predict(self, data: Optional[IBondDataFrame]=None) -> Optional[IBondDataFrame]:
        """
        Model predicting interface. Output predictions for data.

        Args:
            data: ibond dataframe, input training data.

        Returns:
            ibond dataframe: updated dataframe.
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Save parameters to middleware.
        """
        pass

    @abstractmethod
    def load_model(self, model_info: Dict):
        """
        Load model_info into this model.

        Args:
            model_info: the dict representation of the model.
        """
        pass

