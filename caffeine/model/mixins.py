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
#  File name: mixins.py                                                                          
#                                                                                              
#  Create date: 2020/01/04                                                                               
#
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Tuple

from flex.api import make_protocol
import flex.constants as fc

from caffeine.model.base_model import JsonModule, JsonModel
from caffeine.model.base_model import ModelInfo
from caffeine.utils import ClassMethodAutoLog
from caffeine.utils.common_tools import gen_module_id
from caffeine.utils.config import security_config


def get_protocol_var_name(real_name: str):
    for k, v in fc.__dict__.items():
        if v == real_name:
            return k
    return real_name


class ModelSaver(object):
    @ClassMethodAutoLog()
    def make_module(self, params: Optional[Union[List, Dict, BaseModel]]):
        """
        Make module from params.

        Args:
            params: params for the module.
        """
        module_type = f'Module-{self.__class__.__name__}'
        module_id = gen_module_id(prefix=module_type)
        module = JsonModule(
            module_id=module_id,
            module_type=module_type,
            params=params
        )
        return module, module_id

    @ClassMethodAutoLog()
    def save_modules(self,
                     modules: List[JsonModule],
                     federal_info: Dict = {},
                     model_info: Optional[ModelInfo] = None,
                     model_name: Optional[str] = None
                     ) -> str:
        """
        Save modules, i.e. model to json.
        Assume self._context exists.

        Args:
            modules: a list of modules to save.
            federal_info: federal info.
            model_info: optional ModelInfo.
            model_name: optional str, which will appear in model_id.

        Returns:
            str: model id
        """
        model_type = f'{self.__class__.__module__}.{self.__class__.__name__}'
        attributes = None if model_info is None else model_info.model_attributes
        try:
            feat_cols = self.feat_cols
        except:
            self.logger.info(f'Cannot get feat cols for saving model.')
            feat_cols = None
        model = JsonModel(
            model_type=model_type,
            federal_info=federal_info,
            feat_cols = feat_cols,
            modules=modules,
            attributes=attributes
        ).dict()
        self.logger.debug(f'Save model: {model}.')
        if not model_name:
            try:
                model_name = self.algo_name
            except:
                self.logger.error('Cannot find algo_name, default to "model".')
                model_name = 'model'
        model_id = self._context.save_model(
            model,
            # prefix=self.__class__.__name__,
            model_info=attributes,
            #model_name=model_name # NOTE remove model_name in wafer session
        )
        return model_id

    @ClassMethodAutoLog()
    def export_model(self) -> List[Tuple]:
        """
        Export model, i.e. model to json.
        Assume self._context exists.

        Args:
            modules: a list of modules to save.
            federal_info: federal info.

        Returns:
            list of tuple: with key 'model'.
        """
        model_type = f'{self.__class__.__module__}.{self.__class__.__name__}'
        model = JsonModel(
            model_type=model_type,
            federal_info=self.federal_info,
            modules=self.modules
        ).dict()
        return [('model', model)]

    @ClassMethodAutoLog()
    def export_data(self) -> List[Tuple]:
        """
        Export data.

        Args:
            modules: a list of modules to save.
            federal_info: federal info.
            model_info: optional ModelInfo.

        Returns:
            dict: with key 'data' and 'verify_data'.        
        """
        output = [('data', self.train_data),
                  ('data_desc', self.data_desc),
                  ('verify_data', self.verify_data)]
        return output


class FLEXUser(object):
    def init_protocols(self, protocols: List[str]):
        """
        Init FLEX protocols.

        Args:
            protocols: a list of FLEX protocol names.
        """
        for p in protocols:
            self.logger.info(f'Default security config for {p} is: {security_config.get(p, None)}')
        self._protocols = {
            p: make_protocol(
                p,
                self._meta_param['federal_info'],
                self._meta_param.get(
                    'security_param', 
                    {}
                ).get(
                    get_protocol_var_name(p), 
                    security_config.get(p, None)
                ),
                self._meta_param.get('algo_param')
            ) for p in protocols
        }
