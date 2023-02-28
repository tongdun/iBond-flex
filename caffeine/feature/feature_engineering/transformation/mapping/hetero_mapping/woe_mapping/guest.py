from typing import Dict, Optional, Tuple, List, Tuple, Union

import numpy as np
from flex.constants import OTP_PN_FL, IV_FFS

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.utils import label_count
from caffeine.feature.feature_engineering.constant_params import EPS
from caffeine.feature.utils import get_column_data
from .common import HeteroMapToWOEBase


class HeteroMapToWOEGuest(HeteroMapToWOEBase):
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
        if model_info is None and self.use_multiprocess is False:
            self._init_method(IV_FFS)

    @ClassMethodAutoLog()
    def fit(self, data: IBondDataFrame, data_attributes: FeatureDataFrame) -> \
            Tuple[IBondDataFrame, FeatureDataFrame]:          
        label = data.get_y().to_numpy().flatten()   
        good_all_cnt, bad_all_cnt, _, _ = label_count(label) 
        self._model_info = data_attributes.bin_info
        ivs = []

        for i in range(data_attributes.shape):
            col, is_category, is_fillna = data_attributes[i]
            data_col = get_column_data(data, col)
            split_points = self._model_info[col]['split_points']

            good_num, bad_num = self.compute_bin_nums(split_points, is_category, 
                                                is_fillna, data_col, label)

            woe, iv = self.compute_woe_iv(good_num, bad_num, 
                        good_all_cnt, bad_all_cnt)
            is_mono = self.woe_mono(woe, is_category, is_fillna)
            if not is_mono:
                iv = 0.0
                
            self._model_info[col].update({'woe': woe, 'iv': iv})
            ivs.append(iv)

        # woe mapping
        data_attributes['iv'] = ivs
        data_attributes.update_bin(self._model_info)

        data = self.fit_transform(data)
        self.logger.info(f"data_attributes {data_attributes.to_pandas()}")
        self.logger.info(f'model_info {self._model_info}')
        # for host
        self.fit_host(label)
        return data, data_attributes


    @ClassMethodAutoLog()
    def fit_host(self, label: np.ndarray):
        """ 
        WOE mapping for host data.

        Args:
            label, numpy array.
        """
        if self.use_multiprocess:
            max_fea_num = self.p_worker.run_otp_pn_tl_protocol(0, "max", "bin-fea-num", self.federal_info, self._meta_params['security_param'])
        else:
            max_fea_num = self._protocols[OTP_PN_FL].param_negotiate(data=0, param='max', tag='bin-fea-num')
        self.logger.info(f'max_fea_num {max_fea_num}')

        if self.use_multiprocess is False:

            if self.Key is not None:
                self._protocols[IV_FFS].over_loading_first_pub_private_key(self.Key)
                self.logger.info(f'Saved key loaded.')
            else:
                self._protocols[IV_FFS].pre_exchange(label)
                self.logger.info(f'Exchange label done.')

            for i in range(max_fea_num):
                tag = '_'.join([str(self.job_id), str(i)])
                self._protocols[IV_FFS].exchange(label, tag=tag)
        else:
            self.p_worker.run_guest(max_fea_num, label, 
                                    self.federal_info, 
                                    self._meta_params['security_param'], 
                                    self.algo_param,
                                    self.Key
                                )


    @ClassMethodAutoLog()
    def compute_bin_nums(self, points: List, is_category: bool, is_fillna: bool, \
            data_col: np.ndarray, label: np.ndarray) -> Tuple[List[int]]:
        """
        Compute good and bad nums of each bin.

        Args:
            col_info: dict, infos of data_col.
            data_col: np.ndarray, one column of data.
            label: np.ndarray.

        Return:
            good_num: list of int, good num of samples of each bin.
            bad_num: list of int, bad num of samples of each bin.
        """
        good_num, bad_num = [], []
        if is_category:
            for i in range(len(points)):
                idx = np.where(abs(data_col - points[i]) < EPS)[0]
                self.__append_num(good_num, bad_num, idx, label)
        else:
            s = [-np.inf] + points + [np.inf]
            idx_list = [np.where((data_col > s[i]) & (data_col <= s[i+1]))[0] 
                        for i in range(len(s)-1)]
            print(f'******* s {s} {points}')
            for i, index in enumerate(idx_list):
                self.__append_num(good_num, bad_num, index, label)
                print(f'********* good_num {i} {good_num} bad_num {bad_num}')
        if is_fillna is False:
            idx = np.where(np.isnan(data_col))[0]
            self.__append_num(good_num, bad_num, idx, label)

        return good_num, bad_num

    # @ClassMethodAutoLog()
    def __append_num(self, good_num: List, bad_num: List, \
                    idx: Union[List, np.ndarray], label: np.ndarray) \
                    -> None:
        """
        Calculate good_num and bad_num for each idx.

        Args:
            good_num: list.
            bad_num: list.
            idx: list or array, contains index of each bin.
            label: np.ndarray.

        Return:
            None.
        """
        bad_num.append(sum(label[idx]))
        good_num.append(len(idx) - bad_num[-1])

