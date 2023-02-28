from functools import partial
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from flex.constants import HE_DT_FB, OTP_PN_FL
from flex.api import make_protocol

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.utils import get_column_data
from .common import HeteroDtBinBase
from caffeine.feature.feature_engineering.transformation.base_transform import TransformBase


class HeteroDtBinHost(HeteroDtBinBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        Init dtbin base.

        Args: 
            meta_params: dict, containing meta params.
            config: dict, configs for dtbinning.
            model_info: dict, optional and contains model infos for test (transformation).
            e.g.
                meta_params = {

                },
                config = {
                    'equal_num_bin': 50,
                    'map_to_woe': True,
                    'map_to_int': False
                },
                model_info = {
                    'x1': {
                        'split_points': [1, 2, 3],
                        'woe': [1.2, 3.3, 4.8, 5.6],
                        'iv': 0.34,
                        'is_category': False,
                        'is_fillna': True
                    },
                    'x2': {
                        ...
                    },
                    ...
                }
        """
        super().__init__(meta_params, config, context, model_info)
        if model_info is None:
            self._protocols[HE_DT_FB] = make_protocol(
                HE_DT_FB,
                self._meta_params['federal_info'],
                self._meta_params['security_param'].get(HE_DT_FB),
                self.algo_param
            )     


    @ClassMethodAutoLog()
    def fit(self, data: IBondDataFrame, data_attributes: FeatureDataFrame) -> \
            Tuple[IBondDataFrame, FeatureDataFrame]: 
        """
        Binning method for training data.

        Args:
            data: IBondDataFrame, training data.
            data_attributes: FeatureDataFrame, data attributes contains name, is_category, is_fillna.

        Return: 
            data: IBondDataFrame, updated training data.
            data_attributes: FeatureDataFrame, updated attributes.        
        """ 
        data_attributes.sort('is_category')
        fea_num = data_attributes.shape
        max_fea_num = self._protocols[OTP_PN_FL].param_negotiate(fea_num, 'max')       
        ## get encrypted label
        en_label = self._protocols[HE_DT_FB].pre_exchange()
        TransformBase.Label = en_label

        ivs = []
        for i in range(max_fea_num):
            tag = "_".join([str(self.job_id), str(i)])
            self.logger.info(f'>>>>>>>>> local_id {self.local_id}')
            if i >= fea_num:
                self._protocols[HE_DT_FB].exchange(en_label, None, tag=tag)                                                          
                continue            
            
            col, is_category, is_fillna = data_attributes[i]
            data_col = get_column_data(data, col)
            self.logger.info(f'>>>>>>>> processing {col} ...')
            if is_category is False:
                print(f'****** is_category {is_category} col {col} {data_col}')
                split_points = self._protocols[HE_DT_FB].exchange(en_label, data_col, tag=tag)
                print(f'****** after is_category {split_points} col {col}')
                split_points = split_points.tolist()                                            
            else:
                self._protocols[HE_DT_FB].exchange(en_label, None, tag=tag)
                split_points = sorted(set(data_col[~np.isnan(data_col)].tolist()))

            self._model_info[col] = {
                'split_points': split_points, 
                'is_category': is_category, 
                'is_fillna': is_fillna,
            }
            self.logger.info(f'>>>>>>>> Host {self.local_id} has finished {i+1}th features {col}.')
            
        data_attributes.update_bin(self._model_info)
        self.logger.info(f'>>>>>> host {self.local_id} model {self._model_info}')
        self.logger.info(f'data_attributes {data_attributes.to_pandas()}')
        return data, data_attributes
