from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.utils import get_column_data
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.constant_params import feature_params, NUM_DIGITS
from caffeine.feature.feature_engineering.transformation.base_transform import TransformBase
from caffeine.feature.feature_engineering.transformation.binning.param_config import BinConfig


class EquidistBinning(TransformBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params, config: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        EquidistBinning class init.

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
        for i in range(data_attributes.shape):
            col, is_category, _ = data_attributes[i]
            data_col = get_column_data(data, col)
            if is_category is False:
                data_col = sorted(data_col)
                max_value = data_col[-1]
                min_value = data_col[0]
                split_value = np.linspace(min_value, max_value, self.equal_num_bin+1)[1:-1]
                split_value = np.round(split_value, NUM_DIGITS).tolist()
                self._model_info.update({col: {'split_points': split_value, 'is_category': is_category}})
            else:
                self._model_info.update({col: {'split_points': sorted(set(data_col[~np.isnan(data_col)].tolist())), 'is_category': is_category}})

        data_attributes.update_bin(self._model_info)

        return data, data_attributes
