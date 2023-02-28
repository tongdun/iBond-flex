from typing import Dict, Optional, Tuple, List, Tuple
from enum import Enum

import numpy as np
from flex.constants import OTP_PN_FL,  HE_DT_FB, IV_FFS
from flex.api import make_protocol
from pydantic import BaseModel

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.utils.federal_commu import Radio
from caffeine.feature.feature_engineering.constant_params import feature_params, PARALLEL_NUM
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.mixins import FLEXUser, IonicUser
from caffeine.feature.feature_engineering.transformation.mapping.bin_woe import BinWoe
from caffeine.feature.feature_engineering.transformation.base_transform import TransformBase
from caffeine.feature.utils import get_column_data
from caffeine.feature.feature_engineering.transformation.mapping.param_config import WOEConfig
from .parallel_mixin import WOEParallelWorker


class HeteroMapToWOEBase(TransformBase, BinWoe, FLEXUser, IonicUser):
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
        WOEConfig.parse_obj(config)
        self.map_to_woe = config.get('map_to_woe', False)
        self.check_monotone = config.get('check_monotone', True)  
        self.adjust_value_woe = feature_params['adjust_value_woe']
        self.iter = 0

        if model_info is None and self.use_multiprocess is False:
            # num of features to negotiate
            self.init_protocols(
                {
                    OTP_PN_FL: { }
                }
            )
        else:
            self._model_info = model_info

        self.algo_param = dict()
        parallel_num = config.get('min_parallel_num', PARALLEL_NUM)
        self.p_worker = WOEParallelWorker(parallel_num)      

    def _init_method(self, sec):
        self._protocols[sec] = make_protocol(
            sec,
            self.federal_info,
            self._meta_params['security_param'].get(sec),
            {}
        )

    @ClassMethodAutoLog()
    def fit_transform(self, data: IBondDataFrame) -> IBondDataFrame:
        """
        Test method for woe mapping.

        Args:
            data: IBondDataFrame, test data.

        Return:
            data: IBondDataFrame, updated test data after transformation.
        """
        if self.map_to_woe:
            feas = list(self._model_info.keys())
            for col in feas:
                self.logger.info(f'****** fit transform col {col}')
                data_col = get_column_data(data, col)
                data[col] = self.trans_to_woes(data_col, self._model_info[col]['woe'], 
                                                self._model_info[col]['split_points'])
        return data

