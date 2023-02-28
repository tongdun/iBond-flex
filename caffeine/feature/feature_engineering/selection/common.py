import json
import pandas as pd
from logging import getLogger
from pydantic import BaseModel
from re import finditer
from typing import Dict, List, Optional, Tuple

from caffeine.feature.base_model import FeatureAbstractModel
from caffeine.feature.feature_engineering.check_d_type import CheckDTypeCommon
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.utils import make_pipeline
from caffeine.model.base_model import ModelInfo
from caffeine.model.mixins import ModelSaver, JsonModel
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from .utils.hetero_param_config import (HeteroFeatureSelectionMetaParams,
                                        HeteroFeatureSelectionTrainParams, HeteroFeatureSelectionPredictParams,
                                        HeteroSecurity)
from .utils.param_config import (FeatureSelectionMetaParams, FeatureSelectionTrainParams, FeatureSelectionPredictParams)


class FeatureSelectionModel(BaseModel):
    model: Optional[Dict] = None
    selected_x_cols: List
    pipeline: List
    process_method: str
    federal_info: Dict


class FeatureSelectionBase(FeatureAbstractModel, ModelSaver):
    meta_param_model = FeatureSelectionMetaParams
    meta_param_model_dict = {
        'train_param': FeatureSelectionTrainParams,
        'predict_param': FeatureSelectionPredictParams
    }

    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, context: Context, 
                model_info: Optional[Dict]=None):
        """
        Base feature selection class for both hetero and homo cases.

        Args:
            meta_param, a dict contains params like ks_thres, iv_thres and etc..
            e.g.
                meta_param = {
                    'federal_info': {},
                    'd_data': False,
                    'enable_local_auto_selection': False,
                    'process_method': 'hetero',
                    'pipeline': [
                        ('ks': {'ks_thres': 0.01}),
                        ('stepwise': {})
                    ],
                    ...
                }
        """
        super().__init__(meta_params, context, model_info)
        self._parse_meta_params()
        self.logger.info(f"{self.__class__.__name__} meta_param: {meta_params}")

        self._model_info = dict()

        if model_info is not None:
            self.load_model(model_info)
        else:
            make_pipeline(self._meta_params)
            self.logger.info(f"securtiy is {self._meta_params.get('security_param')}")
            self.pipeline = self._meta_params['train_param'].get('pipeline', [])
            self._d_data = self._meta_params.get('d_data', None)
            self.process_method = self._meta_params['train_param'].get('process_method', 'hetero')
            self.logger.info(f">>>> pipeline: {self.pipeline}")

    @ClassMethodAutoLog()
    def _class_init(self, method_classes: Dict, method: str, role: str) -> List:
        """
        Get instance of a specified class by its name.

        Args:
            method_classes: dict, a mapping from str to classname.
            role: str, specify the role.

        Return:
            instance of the SelectionBase class or its subclasses.
        """
        if self.process_method == 'hetero':
            name = f'Hetero{method}Selection{role}'
        else:
            raise NotImplementedError(f"process_methed {self.process_method}is not implemented.")
        class_name = method_classes[self.process_method].get(name, None)
        self.logger.info(f'name: {name}')
        if class_name is None:
            raise RuntimeError(f"No class {class_name} defined.")
        return class_name

    @ClassMethodAutoLog()
    def _report_and_models(self, data_attributes: FeatureDataFrame) -> List:
        """
        Generate report and models.

        Args:
            data_attributes: FeatureDataFrame, contains selection scores.

        Return:
            list, list of models.
        """
        if 'name' in data_attributes.columns:
            self._feat_cols = data_attributes.names
        else:
            self._feat_cols = None
            
        model_infos = self.save_model(data_attributes.to_pandas().to_dict(), self._feat_cols)
        
        data_attributes.add_report('selection_scores', data_attributes.to_pandas().to_dict())
        report = data_attributes.get_report()
        for k, v in report.items():
            self._context.report(k, json.dumps(v))

        return model_infos


    @ClassMethodAutoLog()
    def save_model(self, data_attrib, selected_x_cols):
        """
        Save model to middleware.
        """
        self._model_info['feature_selection'] = dict()
        self._model_info['feature_selection']['d_data'] = self._d_data
        self._model_info['feature_selection']['data_attrib'] = data_attrib
        selection_mod, _ = self.make_module(
            params = FeatureSelectionModel(
                model = self._model_info,
                selected_x_cols = selected_x_cols,
                pipeline = self.pipeline,
                process_method = self.process_method,
                federal_info = self.federal_info
            )
        )

        model_info = ModelInfo(
            None, {}
        )
        model_id = self.save_modules([selection_mod], 
                                    self.federal_info, 
                                    model_info
                                )

        model_info.update_id(model_id)
        return [model_info]

    @ClassMethodAutoLog()
    def load_model(self, model_info: Optional[Dict]=None):
        """
        Load model_info into this model.

        Args:
            model_info: the dict representation of the model.

        -----

        **Example:**

        >>> model_info = {
                'model_type': 'FeatureProcessParticipant',
                'versions': {
                    'caffeine': '0.1',
                    'flex': '1.1',
                    'wafer': '1.0'
                },
                'modules': [
                    {
                        'module_id': 'Module-FeatureProcessParticipant-2020-01-01-10-12-33-12345',
                        'module_type': 'Module-FeatureProcessParticipant',
                        'model_info': {
                            'model': {},
                            'process_method': 'federation_process',
                            'federal_info': {}
                        }
                    }
                ]
            }
        >>> model = FeatureProcessParticipant(meta_params)
        >>> model.load_model(model_info)
        """
        self.logger.info(f'Parsing model_info: {model_info}...')
        model_info = JsonModel.parse_obj(model_info)
        
        self._meta_params = model_info.modules[0].params
        self.logger.info(f'Loading _meta_params: {self._meta_params}...')

        model = model_info.modules[0].params['model']
        self.logger.info(f'Loading model: {model}...')
        self._model_info = model

        process_method = model_info.modules[0].params['process_method']
        self.logger.info(f'Loading process_method: {process_method}...')
        self.process_method = process_method

        selected_x_cols = model_info.modules[0].params['selected_x_cols']
        self.logger.info(f'Loading selected_x_cols: {selected_x_cols}...')
        self.selected_x_cols  = selected_x_cols

        pipeline = model_info.modules[0].params['pipeline']
        self.logger.info(f'Loading pipeline: {pipeline}...')
        self.pipeline = pipeline

        self._d_data = model['feature_selection']['d_data']
        self._meta_params['d_data'] = self._d_data
        self.logger.info(f'Loading _d_data: {self._d_data}...')

        self.logger.info(f'Model loaded.')


class HeteroFeatureSelectionBase(FeatureSelectionBase):
    meta_param_model = HeteroFeatureSelectionMetaParams
    meta_param_model_dict = {
        'train_param': HeteroFeatureSelectionTrainParams,
        'predict_param': HeteroFeatureSelectionPredictParams,
        'security_param': HeteroSecurity
    }

