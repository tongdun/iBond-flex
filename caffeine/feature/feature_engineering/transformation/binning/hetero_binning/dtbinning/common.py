from typing import Dict, Optional, Tuple, List, Tuple

import numpy as np
from flex.constants import OTP_PN_FL

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.utils.federal_commu import Radio
from caffeine.feature.feature_engineering.constant_params import feature_params
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.mixins import FLEXUser, IonicUser
from ...local_binning.equifrequent_bin import EquifrequentBinning
from caffeine.feature.feature_engineering.transformation.binning.param_config import BinConfig


class HeteroDtBinBase(EquifrequentBinning, FLEXUser, IonicUser):
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
        BinConfig.parse_obj(config)
        self.max_bin_num = config.get('max_bin_num', feature_params['max_bin_num'])
        self.min_bin_num = config.get('min_bin_num', feature_params['min_bin_num'])
        self.check_monotone = config.get('check_monotone', True)
        self.bin_ratio = config.get('bin_ratio', feature_params['bin_ratio'])
        self.algo_param = {
                        'bin_ratio': self.bin_ratio,
                        'max_bin_num': self.max_bin_num,
                        'min_bin_num': self.min_bin_num,
                        'frequent_value': self.equal_num_bin,
                        'host_party_id': self.hosts[0]
                    }

        if model_info is None:
            # num of features to negotiate
            self.init_protocols(
                {
                    OTP_PN_FL: { },
                }
            )
        else:
            self._model_info = model_info

    @ClassMethodAutoLog()
    def _get_index(self, data: np.ndarray, node: float) -> Tuple[list]:
        """
        Get left and right index of the current leaf node.

        Args:
            data: np.ndarray, subset data at the current node.
            node: float, current leaf node.

        Return:
            left_idx: list.
            right_idx: list.
        """
        right_idx = np.where(data > node)[0]
        left_idx = np.where(data <= node)[0]
        return left_idx, right_idx
