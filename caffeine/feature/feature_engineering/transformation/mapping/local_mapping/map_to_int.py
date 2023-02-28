import numpy as np
import pandas as pd
from logging import getLogger
from pydantic import BaseModel
from typing import Dict, Tuple, Optional, Union, List

from caffeine.feature.feature_engineering.constant_params import feature_params, EPS
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.transformation.base_transform import TransformBase
from caffeine.feature.utils import get_column_data
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame


class IntegerMapConfig(BaseModel):
    map_to_int: bool = True

    class Config:
        schema_extra = {
            'expose': ["map_to_int"]
        }


class MapToInt(TransformBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Dict, context, model_info: Optional[Dict]=None):
        """
        Base bin class init.

        Args: 
            meta_params: dict, contains federal_info, transform_pipeline and its configs.
            config: dict, contains algorighm params for transformation, probably for flex.
            e.g. meta_params = {
                    'federal_info': {},
                    'process_method': 'hetero',
                    ....
            }
            configs = {
                {'map_to_int': True}
            }
            context: Context, context of the model to save models etc..
                e.g. a wafer session.
            model_info: dict, saved model for prediction.
        """
        super().__init__(meta_params, config, context, model_info)
        IntegerMapConfig.parse_obj(config)
        self.map_to_int = config.get('map_to_int', True)
        self.logger.info(f'***** map_to_int {self.map_to_int}')
        if model_info is None:
            self._model_info = dict()
        else:
            self._model_info = model_info

    @ClassMethodAutoLog()
    def fit(self, data: Optional[IBondDataFrame]=None, 
                  data_attributes: Optional[FeatureDataFrame]=None) -> \
                  Tuple[IBondDataFrame, FeatureDataFrame]: 
        """
        Train method for mapping to integer.

        Args:
            data: IBondDataFrame, training data.
            data_attributes: FeatureDataFrame, including name, is_category, is_fillna.

        Return:
            data, IBondDataFrame, updated data.
            data_attributes: FeatureDataFrame, updated FeatureDataFrame 
                after binning.
        """
        self._model_info = data_attributes.bin_info
        data = self.fit_transform(data)
        self.logger.info(f'***** data {data}')
        return data, data_attributes

    @ClassMethodAutoLog()
    def fit_transform(self, data: IBondDataFrame) -> IBondDataFrame:
        """
        Test method for mapping to integer.

        Args:
            data: IBondDataFrame, test data.

        Return:
            data: IBondDataFrame, updated test data after transformation.
        """
        if self.map_to_int:
            feas = list(self._model_info.keys())
            if len(feas) > 0:
                split_points = []
                for col in feas:
                    split_points.append(self._model_info[col]['split_points'])

                data_numpy = data.select(feas).to_numpy()
                data = data.drop(feas)
                new_data = np.array(list(map(lambda x, y: np.digitize(x, y, right=True), data_numpy.T, split_points))).T
                self.logger.info(f"new_data shape {new_data.shape}")
                data[feas] = new_data
            
        return data