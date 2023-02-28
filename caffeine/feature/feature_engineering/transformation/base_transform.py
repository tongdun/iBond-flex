from logging import getLogger
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.constant_params import feature_params, USE_MULTIPROCESS


class TransformBase(object):
    Label = None
    Key = None

    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        Base transform class init.

        Args: 
            meta_params: dict, contains federal_info, transform_pipeline and its configs.
                e.g. meta_params = {
                        'federal_info': {},
                        'process_method': 'hetero',
                        ....
                }
            config: dict, contains algorighm params for transformation, probably for flex.
            context: Context, context of the model to save models etc..
                e.g. a wafer session.
            model_info: dict, saved model for prediction.
        """
        self.logger = getLogger(self.__class__.__name__)
        self._meta_params = meta_params
        self._context = context
        self.logger.info(f">>>>> federal_info {meta_params.get('federal_info')}")
        if model_info is None:
            common_params = meta_params['train_param'].get('common_params')
            self.logger.info(f"******* meta_params {meta_params} ")
            self.down_feature_num = common_params.get('down_feature_num', feature_params['down_feature_num'])
            self.use_multiprocess = common_params.get('use_multiprocess', USE_MULTIPROCESS)

    def fit(self, data: Optional[IBondDataFrame]=None, 
                  data_attributes: Optional[FeatureDataFrame]=None) -> \
                  Tuple[IBondDataFrame, FeatureDataFrame]: 
        """ 
        Fit method for training data.
        Args:
            data: ibond dataframe.
            data_attributes: feature dataframe, for storing feature infos.

        Return:
            ibond dataframe, after fitting.
            feature dataframe, after fitting.
        """
        return data, data_attributes

    def fit_transform(self, data: Optional[IBondDataFrame]) -> Optional[IBondDataFrame]:
        """ 
        Fit transform method for prediction.
        Args:
            data: ibond dataframe.

        Return:
            ibond dataframe, after transformation.
        """
        return data