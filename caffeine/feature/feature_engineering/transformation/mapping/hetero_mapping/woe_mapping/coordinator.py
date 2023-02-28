from typing import Dict, Optional, Tuple, List, Tuple

import numpy as np
from flex.constants import OTP_PN_FL

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from .common import HeteroMapToWOEBase


class HeteroMapToWOECoord(HeteroMapToWOEBase):
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

    @ClassMethodAutoLog()
    def fit(self, data: IBondDataFrame, data_attributes: FeatureDataFrame) -> \
            Tuple[IBondDataFrame, FeatureDataFrame]:         
        """
        WOE mapping for training data at coordinator.

        Args:

        """  
        if self.use_multiprocess:
            max_fea_num = self.p_worker.run_otp_pn_tl_protocol(None, "max", "bin-fea-num", self.federal_info, self._meta_params['security_param'])
            self.p_worker.run_coord(max_fea_num, self.federal_info, self._meta_params['security_param'])
        else:
            max_fea_num = self._protocols[OTP_PN_FL].param_negotiate(data=None, param='max', tag='bin-fea-num')
        
        return data, data_attributes
        