from typing import Dict, Tuple, List, Optional, Union
from logging import getLogger

import pandas as pd
import numpy as np
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.utils import get_column_data
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.constant_params import feature_params, NUM_DIGITS
from caffeine.feature.feature_engineering.transformation.base_transform import TransformBase
from caffeine.feature.feature_engineering.transformation.binning.param_config import BinConfig


class EquifrequentBinning(TransformBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        EquifrequentBinning class init.

        Args: 
            meta_params: dict, contains federation, transform_pipeline and its configs.
            config: dict, contains algorighm params for transformation, probably for flex.
            e.g. meta_params = {
                    'federal_info': {},
                    'process_method': 'hetero',
                    ....
            }
            configs = {
                {'equal_num_bin': 10, 'map_to_int': True}
            }
        """
        super().__init__(meta_params, config, context, model_info)
        BinConfig.parse_obj(config)
        self.equal_num_bin = config.get('equal_num_bin', feature_params['equal_num_bin'])
        
        if model_info is None:
            self._model_info = dict()
        else:
            self._model_info = model_info

    @ClassMethodAutoLog()
    def fit(self, data: IBondDataFrame, data_attributes: FeatureDataFrame) -> \
            Tuple[IBondDataFrame, FeatureDataFrame]:        
        """
        Transformation method for training data.

        Args:
            data: IBondDataFrame, training data.
            data_attributes: FeatureDataFrame, including name, is_category, is_fillna.

        Return:
            data, IBondDataFrame, updated data.
            data_attributes: FeatureDataFrame, updated FeatureDataFrame 
                after binning.
        """
        # bin threshold of feature
        for i in range(data_attributes.shape):
            col, is_category, _ = data_attributes[i]
            data_col = get_column_data(data, col)
            split_points = self.fit_one(data_col, is_category)
            self._model_info.update({col: {'split_points': split_points, 'is_category': is_category}})

        data_attributes.update_bin(self._model_info)

        return data, data_attributes

    @ClassMethodAutoLog()
    def fit_one(self, data: Union[np.ndarray, pd.Series], is_category: bool) -> List:  
        """
        Equifreqeunt binning for one column data.

        Args:
            data: np.ndarray or pd.Series, column data.
            data_attributes: FeatureDataFrame, including name, is_category, is_fillna.

        Return:
            data, IBondDataFrame, updated data.
            data_attributes: FeatureDataFrame, updated FeatureDataFrame 
                after binning.
        """
        if is_category is False:
            # use num_bin to do equifrequent binning, must be less than
            # unique number of data.
            self.num_bin = min(self.equal_num_bin, len(np.unique(data)))
            # self.num_bin = self.equal_num_bin
            percentiles = np.linspace(0, 1, self.num_bin + 1)[1:-1]*100
            percent_values = np.percentile(data[~np.isnan(data)], percentiles)
            percent_values = sorted(set(np.round(percent_values, NUM_DIGITS)))                
            return percent_values
        else:
            return sorted(set(data[~np.isnan(data)].tolist()))

