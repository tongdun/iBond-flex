from typing import Dict, Tuple, Optional, Union, List
from logging import getLogger
from collections import defaultdict
from enum import Enum

import pandas as pd
import numpy as np
from scipy import sparse
from pydantic import BaseModel, Field

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.utils.dataframe import parse_ibonddf
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.transformation.base_transform import TransformBase
from caffeine.feature.utils import get_column_data
from caffeine.feature.common_tools import onehot_encoding, array_hstack


ONEHOTEMBED = 'OneHotEmbed'
STREMBED = 'StrEmbed'

class OneHotMapConfig(BaseModel):
    category_to_onehot: bool = True
    continuous_to_onehot: bool = False
    to_sparse: bool = False

class OneHotMap(TransformBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        OneHot mapping class init.

        Args: 
            meta_params: dict, contains federal_info, transform_pipeline and its configs.
            config: dict, contains algorighm params for transformation, probably for flex.
            e.g. meta_params = {
                    'federal_info': {},
                    'process_method': 'hetero',
                    ....
            }
            configs = {
                {'to_sparse': True, 'continuous_to_onehot': True, 'category_to_onehot': True}
            }
        """
        super().__init__(meta_params, config, context, model_info)
        OneHotMapConfig.parse_obj(config)
        self.to_sparse = config.get('to_sparse')
        self.continuous_to_onehot = config.get('continuous_to_onehot')
        self.category_to_onehot = config.get('category_to_onehot')
        if model_info is None:
            self._model_info = dict()
        else:
            self._model_info = model_info

    @ClassMethodAutoLog()
    def fit(self, data: Optional[IBondDataFrame]=None, 
                  data_attributes: Optional[FeatureDataFrame]=None) -> \
                  Tuple[IBondDataFrame, FeatureDataFrame]: 
        """
        Train method for mapping to onehot encoding.

        Args:
            data: IBondDataFrame, training data.
            data_attributes: FeatureDataFrame, including name, is_category, is_fillna.

        Return:
            data, IBondDataFrame, updated data.
            data_attributes: FeatureDataFrame, updated FeatureDataFrame 
                after binning.
        """
        if not self.continuous_to_onehot and not self.category_to_onehot:
            self.logger.info("No onehot encoding performed!")
            return data, data_attributes

        rm_cols = []
        onehot_columns = []

        if not self.to_sparse:
            onehot_array = np.empty(shape=[data.shape[0],0])
        else:
            onehot_array = sparse.coo_matrix((data.shape[0], 0))

        for i in range(data_attributes.shape):

            col, is_category, _ = data_attributes[i]
            if not self.continuous_to_onehot and not is_category:
                continue 
            if not self.category_to_onehot and is_category:
                continue
            if is_category:
                ls =  col.split('_')
                if STREMBED in ls and ls[-2]==STREMBED:
                    continue
                uniq_vals = list(range(len(data_attributes.bin_info[col].get('split_points'))))
            else:
                uniq_vals = list(range(len(data_attributes.bin_info[col].get('split_points'))+1))
                
            data_col = get_column_data(data, col)
            self.logger.info(f'***** uniqvals {is_category} {uniq_vals} data_col {data_col}')
            tmp, names = onehot_encoding(data_col, uniq_vals, col+'_'+ONEHOTEMBED, to_sparse=self.to_sparse)
            onehot_columns.extend(names)
            onehot_array = array_hstack([onehot_array, tmp])
            rm_cols.append(col)
            
        data = self._update_dataframe(data, rm_cols, onehot_array, onehot_columns)
        
        desc = parse_ibonddf(data)
        feat_cols = desc["feat_cols"]

        names = np.setdiff1d(feat_cols, data_attributes.names).tolist()
        data_attributes = data_attributes.select_by_name(feat_cols)
        new_attributes = pd.DataFrame({'names': names, 'is_category': True, 'is_fillna': True})
        data_attributes.append(new_attributes)

        return data, data_attributes

    @ClassMethodAutoLog()
    def _update_dataframe(self, data, rm_cols, onehot_array, onehot_columns):
        data = data.drop(rm_cols)
        desc = data.data_desc

        if not self.to_sparse:
            data[onehot_columns] = onehot_array
        
        else:
            new_data = pd.DataFrame.sparse.from_spmatrix(onehot_array, columns=onehot_columns)
            data = pd.concat((data.to_pandas(), new_data), axis=1)
            data = self._context.create_dataframe(data, desc, to_sparse=self.to_sparse)

        return data


    @ClassMethodAutoLog()
    def fit_transform(self, data: IBondDataFrame) -> IBondDataFrame:
        """
        Test method for onehot mapping.

        Args:
            data: IBondDataFrame, test data.

        Return:
            data: IBondDataFrame, updated test data after transformation.
        """
        onehot_columns = []
        rm_cols = []

        if not self.to_sparse:
            onehot_array = np.empty(shape=[data.shape[0],0])
        else:
            onehot_array = sparse.coo_matrix((data.shape[0], 0))

        for col, v in self._model_info.items():

            data_col = get_column_data(data, col)
            is_category = v['is_category']
            if not self.continuous_to_onehot and not is_category:
                continue 
            if not self.category_to_onehot and is_category:
                continue
            if is_category:
                ls =  col.split('_')
                if STREMBED in ls and ls[-2]==STREMBED:
                    continue
                uniq_vals = list(range(len(v['split_points'])))
            else:
                uniq_vals = list(range(len(v['split_points'])+1))

            tmp, new_names = onehot_encoding(data_col, uniq_vals, col+'_'+ONEHOTEMBED, to_sparse=self.to_sparse) 
            onehot_columns.extend(new_names)
            onehot_array = array_hstack([onehot_array, tmp])
            rm_cols.append(col)

        data = self._update_dataframe(data, rm_cols, onehot_array, onehot_columns)
        return data

