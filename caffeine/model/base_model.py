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
#  File name: base_model.py                                                                         
#                                                                                              
#  Create date: 2020/11/24                                                                               
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
from caffeine.utils.convergence import converge_func_factory
from caffeine.utils.dataframe import parse_ibonddf


class ModelInfo(object):
    def __init__(self, model_id: str, model_attributes: Dict[str, Dict]):
        """
        The information of a saved model.

        Args:
            model_id: str, the saved model id returned from the middelware.
            model_attributes: dict of string, keys are strings, e.g. metrics.

        -----

        **Examples:**

        >>> model_info = ModelInfo(
                model_id = '123',
                model_attributes = {
                    'metrics': {
                        'ks': 0.35
                    }
                }
            )
        >>> model_info.model_id
        '123'

        """
        self._model_id = model_id
        self._model_attributes = model_attributes

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_attributes(self) -> Dict[str, Dict]:
        return self._model_attributes

    def update_attributes(self, new_attributes: Dict[str, Dict]):
        self._model_attributes.update(
            new_attributes
        )

    def update_id(self, model_id: str):
        self._model_id = model_id

    def __repr__(self):
        return f'''
-----------------
-model id: 
{self.model_id}
-model attributes: 
{self.model_attributes}
-----------------
'''


class MetaParams(BaseModel):
    train_param: Any
    predict_param: Any
    security_param: Any
    federal_info: Any


class AbstractModel(metaclass=ABCMeta):
    meta_param_model = MetaParams
    meta_param_model_dict = {
        'train_param': BaseModel,
        'predict_param': BaseModel
    }

    @abstractmethod
    def __init__(self, meta_param: Dict, context: Context):
        """
        Initiate Model.

        Args:
            meta_param: a dict of meta parameters.
            context: Context, context of the model to save models etc..
                e.g. a wafer session.
        """
        self.logger = getLogger(self.__class__.__name__)
        self._meta_param = meta_param
        self._context = context

        self._train_data_info = {}
        self._feat_cols = None

    def _parse_meta_param(self):
        if self.meta_param_model is not None:
            try:
                self._meta_param = self.meta_param_model.parse_obj(
                    self._meta_param
                ).dict(exclude_none=True)
                self.logger.info(f'Parsed meta_param {self._meta_param}')
            except:
                self.logger.error(f'!!!Parse meta_param {self._meta_param} failed!!!')
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

    def _check_early_stop_param(self):
        if 'train_param' in self._meta_param:
            train_param = self._meta_param['train_param']
            self.early_stop_param = train_param.get('early_stop_param', {'early_stop': False, 'early_stop_step': 'epoch', 'early_stop_method': 'abs', 'early_stop_eps': 1e-5})
            self.converge_flag = False
            early_stop_param_list = self.early_stop_param.keys()
            if 'early_stop' not in early_stop_param_list or self.early_stop_param['early_stop'] not in [True, False]:
                self.logger.info(f'early_stop maybe not config or config wrong, now set default False')
                self.early_stop_param['early_stop'] = False

            if self.early_stop_param['early_stop'] is True:
                if 'early_stop_step' not in early_stop_param_list or self.early_stop_param['early_stop_step'] not in ['iter', 'epoch']:
                    self.logger.info(f'early_stop_step current only support iter or epoch')
                    self.logger.info(f'early_stop_step not config or config wrong, now set default epoch')
                    self.early_stop_param['early_stop_step'] = 'epoch'
                if 'early_stop_method' not in early_stop_param_list or self.early_stop_param['early_stop_method'] not in ['abs', 'diff']:
                    self.logger.info(f'early_stop_method current only support abs or diff')
                    self.logger.info(f'early_stop_method not config or config wrong, now set default abs')
                    self.early_stop_param['early_stop_step'] = 'abs'
                if 'early_stop_eps' not in early_stop_param_list:
                    self.logger.info(f'early_stop_eps not config, now set default 1e-5')
                    self.early_stop_param['early_stop_eps'] = 1e-5

                self.converge_func = converge_func_factory(early_stop_method=self.early_stop_param['early_stop_method'], tol=self.early_stop_param['early_stop_eps'])

    def _before_train(
        self, 
        train_data: Optional[IBondDataFrame] = None, 
        val_data: Optional[IBondDataFrame] = None
    ):
        if train_data:
            self._train_data_info.update(
                parse_ibonddf(train_data)
            )
            self._feat_cols = self._train_data_info['feat_cols']

    def _after_train(
        self, 
        train_data: Optional[IBondDataFrame] = None, 
        val_data: Optional[IBondDataFrame] = None,
        model_infos: Optional[List] = None
    ):
        pass

    @abstractmethod
    def train(self, train_data: Optional[IBondDataFrame], val_data: Optional[IBondDataFrame]):
        """
        Model training interface. Train via data, update self model
        parameters.

        Args:
            train_data: optional ibond dataframe, input training data.
            val_data: optional ibond dataframe, input validation data.
        """
        pass

    @abstractmethod
    def predict(self, data: Union[None, IBondDataFrame],
                predict_id: Optional[str] = '*') -> Union[None, IBondDataFrame]:
        """
        Model predicting interface. Output predictions for data.

        Args:
            data: ibond dataframe, input training data.

        Returns:
            ibond dataframe: output predictions.
        """
        pass

    @abstractmethod
    def _save_params(self):
        """
        Save parameters to middleware.
        """
        pass

    @abstractmethod
    def load_params(self, params: Dict):
        """
        Load parameters into this model.

        Args:
            params: the dict representation of the model.
        """
        pass


class JsonVersions(BaseModel):
    caffeine: str = config.version
    # TODO
    flex: str = '1.2dev'
    wafer: str = config.middleware_version


class JsonModule(BaseModel):
    module_id: str
    module_type: str
    params: Optional[Union[List, Dict, BaseModel]]


class JsonModel(BaseModel):
    model_type: str
    federal_info: Optional[Dict]
    versions: JsonVersions = JsonVersions()
    feat_cols: Optional[List[str]]
    attributes: Optional[Dict]
    modules: List[JsonModule] = []


if __name__ == '__main__':
    empty_model = JsonModel(
        model_type = 'Empty',
        #versions = JsonVersions(),
        modules = [
        ]
    )
    print(empty_model.json())
