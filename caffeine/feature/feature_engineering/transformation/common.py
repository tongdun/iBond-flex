import json
import pandas as pd
from logging import getLogger
from pydantic import BaseModel
from re import finditer
from typing import Dict, Tuple, Optional, List

from caffeine.feature.base_model import FeatureAbstractModel
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.utils import make_pipeline
from caffeine.model.base_model import ModelInfo
from caffeine.model.mixins import ModelSaver, JsonModel
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from .param_config import (FeatureTransformMetaParams, FeatureTransformPredictParams,
                           FeatureTransformTrainParams, HeteroFeatureTransformMetaParams,
                           HeteroFeatureTransformTrainParams,
                           LocalFeatureTransformTrainParams,
                           LocalFeatureTransformMetaParams,
                           HeteroSecurity
                           )


class FeatureTransformationInfo(BaseModel):
    model: Optional[Dict]=None
    pipeline: List[Tuple]
    federal_info: Dict
    process_method: str

class FeatureTransformBase(FeatureAbstractModel, ModelSaver):
    meta_param_model = FeatureTransformMetaParams
    meta_param_model_dict = {
        'train_param': FeatureTransformTrainParams,
        'predict_param': FeatureTransformPredictParams
    }

    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        Init transform base.

        Args:
            meta_params: dict, contains federal_info, transform_pipeline and its configs.
                e.g. meta_params = {
                        'transform_pipeline': [
                            ('equifrequent_bin': {'equal_num_bin': 10, 'is_mapping': True}),
                            ....
                        ],
                        'federal_info': {},
                        ....
                }
            context: Context, context of the model to save models etc..
                e.g. a wafer session.
            model_info: dict, saved model for prediction.
        """
        super().__init__(meta_params, context, model_info)
        self._parse_meta_params()

        if model_info is not None:
            self.load_model(model_info)
        else:
            make_pipeline(self._meta_params)
            self._d_data = self._meta_params.get('d_data', None)
            self.process_method = self._meta_params['train_param'].get('process_method')
            self.pipeline = self._meta_params['train_param'].get('pipeline')
            self.logger.info(f'********pipeline: {self.pipeline}')

    @ClassMethodAutoLog()
    def _report_and_models(self, data_attributes: FeatureDataFrame) -> List:
        """
        Generate report and models.

        Args:
            data_attributes: FeatureDataFrame, contains selection scores.

        Return:
            list, list of models.
        """
        # save model
        if 'name' in data_attributes.columns:
            self._feat_cols = data_attributes.names
        else:
            self._feat_cols = None
        model_info = dict()
        model_info['feature_transform'] = data_attributes.bin_info
        model_info['data_attrib'] = data_attributes.to_pandas().to_dict()
        model_infos = self.save_model(model_info)

        data_attributes.add_report('transform_info', data_attributes.bin_info)
        report = data_attributes.get_report()
        for k, v in report.items():
            self._context.report(k, json.dumps(v))

        return model_infos

    def save_model(self, model_info: Dict):
        """
        Save model to middleware.
        """
        self._model_info = model_info
        self.params = {
            'pipeline': self.pipeline,
            'process_method': self.process_method,
            'federal_info': self._meta_params['federal_info'],
            'model': self._model_info,
        }
        fea_trans_mod, _ = self.make_module(
            params = FeatureTransformationInfo(
                model = self.params['model'],
                process_method = self.params['process_method'],
                federal_info = self.params['federal_info'],
                pipeline = self.params['pipeline'],
            )
        )

        model_info = ModelInfo(
            None, {}
        )
        model_id = self.save_modules([fea_trans_mod], 
                                    self.params['federal_info'], 
                                    model_info
                                )

        model_info.update_id(model_id)
        return [model_info]

    @ClassMethodAutoLog()
    def load_model(self, model_info: Dict):
        """
        Load model_info into this model.

        Args:
            model_info: the dict representation of the model.

        -----

        **Example:**

        >>> model_info = {
                'model_type': 'FeatureEngineeringGuest',
                'versions': {
                    'caffeine': '0.1',
                    'flex': '1.1',
                    'wafer': '1.0'
                },
                'modules': [
                    {
                        'module_id': 'Module-FeatureEngineeringGuest-2020-01-01-10-12-33-12345',
                        'module_type': 'Module-FeatureEngineeringGuest',
                        'model_info': {
                            'model': {},
                            'process_method': 'hetero',
                            'federal_info': {},
                            'feature_engineering_pipeline': {
                                'transform_pipeline': [
                                    ('equifrequent_bin': {'equal_num_bin': 10, 'map_to_int': True}),
                                    ....
                                ],
                                'pipeline': [
                                    ('ks': {'ks_thres': 0.01}),
                                    ('stepwise': {}),
                                    ...
                                ]
                            },
                        }
                    }
                ]
            }
        >>> model = FeatureEngineeringGuest(meta_params)
        >>> model.load_model(model_info)
        """
        self.logger.info(f'Parsing model_info: {model_info}...')
        model_info = JsonModel.parse_obj(model_info)

        self._meta_params = model_info.modules[0].params
        self.logger.info(f'Loading _meta_params: {self._meta_params}...')

        model = self._meta_params['model']
        self.logger.info(f'Loading model: {model}...')
        self._model_info = model

        self.pipeline = self._meta_params.get('pipeline')
        self.logger.info(f'Loading pipeline: {self.pipeline}...')

        federal_info = self._meta_params['federal_info']
        self.logger.info(f'Loading federal_info: {federal_info}...')
        self._meta_params['federal_info'] = federal_info

        process_method = self._meta_params['process_method']
        self.logger.info(f'Loading process_method: {process_method}...')
        self.process_method = process_method

        self.logger.info(f'Model loaded.')


class HeteroFeatureTransformBase(FeatureTransformBase):
    meta_param_model = HeteroFeatureTransformMetaParams
    meta_param_model_dict = {
        'train_param': HeteroFeatureTransformTrainParams,
        'predict_param': FeatureTransformPredictParams,
        'security_param': HeteroSecurity
    }

class LocalFeatureTransformBase(FeatureTransformBase):
    meta_param_model = LocalFeatureTransformMetaParams
    meta_param_model_dict = {
        'train_param': LocalFeatureTransformTrainParams,
        'predict_param': FeatureTransformPredictParams,
        'security_param': None
    }
