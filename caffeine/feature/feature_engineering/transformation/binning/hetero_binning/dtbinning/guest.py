from functools import partial
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from flex.constants import HE_DT_FB, OTP_PN_FL
from flex.api import make_protocol

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.utils import partial_map, label_count
from caffeine.feature.feature_engineering.constant_params import DEFAULT_MAX, EPS
from .common import HeteroDtBinBase
from caffeine.feature.feature_engineering.transformation.base_transform import TransformBase


class HeteroDtBinGuest(HeteroDtBinBase):
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
        self.node_num = int(self.bin_ratio * data.shape[0])
        self.logger.info(f'>>>> start guest calculation')
        label = data.get_y().to_numpy().flatten()    
        good_all_cnt, bad_all_cnt, _, _ = label_count(label)

        ivs = []
        # binning, woe and iv
        for i in range(data_attributes.shape):
            col, is_category, is_fillna = data_attributes[i]
            data_col = data[col].to_numpy().flatten()
            self.logger.info(f'****** processing col {col}')
            if is_category is False:
                split_points = self._get_bestSplit_list_local(label, data_col)
            else:
                split_points = sorted(set(data_col[~np.isnan(data_col)].tolist()))

            self._model_info[col] = {
                'split_points': split_points, 
                'is_category': is_category, 
                'is_fillna': is_fillna,
            }

        data_attributes.update_bin(self._model_info)
        self.logger.info(f'********** data_attributes {data_attributes.to_pandas()}')

        # binning for host
        self.fit_host(label)
        return data, data_attributes


    @ClassMethodAutoLog()
    def fit_host(self, label: np.ndarray) -> None:
        """
        Training method of binning for host.
        """
        # Step1: get max num of features to deal with
        max_fea_num = self._protocols[OTP_PN_FL].param_negotiate(0, 'max')
        # Step2: broadcast encrypted label
        self._protocols[HE_DT_FB].pre_exchange(label)
        TransformBase.Key = self._protocols[HE_DT_FB].save_first_pub_private_key()

        # Step3: split index and ivs
        for i in range(max_fea_num):
            tag = '_'.join([str(self.job_id), str(i)])
            self._protocols[HE_DT_FB].exchange(label, tag=tag)

    @ClassMethodAutoLog()
    def _get_bestSplit_list_local(self, label: np.ndarray, data: np.ndarray) -> List[float]:
        """
        Entrance of guest split points.

        Args:
            label: np.ndarray, label info.
            data: np.ndarray, input column data.

        Return:
            split points: list.
        """
        # equifrequent binning first
        split_list = []
        self.initial_bin = super().fit_one(data, False)
        if len(self.initial_bin) < self.max_bin_num or len(self.initial_bin) == 0:
            return self.initial_bin

        self._get_bestSplit_list_step_local(data, label, split_list)
        return sorted(split_list)

    @ClassMethodAutoLog()
    def _get_bestSplit_list_step_local(self, sample_set: np.ndarray, label: np.ndarray, split_list: List) -> None:
        """
        Main method for dtbin calculation.

        Args:
            sample_set: np.ndarray, input column data.
            label: np.ndarray, label.
            split_list: list, split points for sample_set.

        Return:
            None.
        """
        if len(split_list) >= self.max_bin_num - 1:
            return

        bin_info = np.apply_along_axis(self._get_bin_index, 1, 
                                       np.array(self.initial_bin).reshape(-1,1), 
                                       sample_set, 
                                       label
                                    )
        bin_info = bin_info[np.where(np.min(bin_info[:, 1:3], axis=1) != 0)[0]]
        split = self._choose_best_split_local(bin_info[:, 1:])
        self.logger.info(f'>>>> split {split}')
        if split is not None:
            split_list.append(float(bin_info[split, 0]))
            if len(split_list) >= self.max_bin_num - 1:
                return

            idx_left, idx_right = self._get_index(sample_set, bin_info[split, 0])
            sample_left, sample_right = sample_set[idx_left], sample_set[idx_right]
            label_left, label_right = label[idx_left], label[idx_right]

            if sample_left.shape[0] >= self.node_num * 2:
                self._get_bestSplit_list_step_local(sample_left, label_left, split_list)
            if sample_right.shape[0] >= self.node_num * 2:
                self._get_bestSplit_list_step_local(sample_right, label_right, split_list)

    @ClassMethodAutoLog()
    def _choose_best_split_local(self, bin_info: np.ndarray) -> Union[int, None]:
        """
        Choose best splits and return index.

        Args:
            bin_info: np.ndarray, contains good and bad nums of each split points.

        Return:
            index of the split point corresponding to the mininum gini value. 
            int -- index, None -- no solution.
        """
        split = None
        if len(bin_info) == 0 or bin_info is None:
            return split

        num_samples, bad_samples = sum(bin_info[0, :2]), sum(bin_info[0, 2:])
        gini = self.__gini(bad_samples, num_samples)
        if gini < EPS:
            return split

        gini_values = np.apply_along_axis(self.__leaf_gini, 1, bin_info, num_samples).tolist()   
        mini_gini = min(gini_values)
        self.logger.info(f'>>>>>>> gini_values {gini} {gini_values}')
        if mini_gini < gini:
            min_idx = gini_values.index(mini_gini)
            return min_idx

        return split

    @ClassMethodAutoLog()
    def _get_bin_index(self, node: float, data: np.ndarray, label: np.ndarray):
        """
        Get bad nums and counts of left and right side of each node.

        Args:
            data: np.ndarray, subset data corresponding to node.
            label: np.ndarray, subset label corresponding to node.
            node: float, current node point.

        Return:
            list of info, [ count of left side of node,
                            count of right side of node,
                            bad_count of left side of node,
                            bad_count of right side of node ]
        """
        left_idx, right_idx = self._get_index(data, node)
        right_label = label[right_idx]
        left_label = label[left_idx]
        return [node, left_label.shape[0], right_label.shape[0], np.sum(left_label), np.sum(right_label)]

    @ClassMethodAutoLog()
    def __leaf_gini(self, leaf_info: np.ndarray, num_samples: int):
        """
        Leaf node gini calculation.

        Args:
            leaf_info: np.ndarray, [ cnt of left, cnt of right, bad_cnt of left, bad_cnt of right] 
                of each leaf node.
            num_samples: int, number of samples in the father dataset.

        Return:
            gini: float, gini value of each leaf node.
        """
        cnt = leaf_info[:2]
        bad = leaf_info[2:]
        ratio = cnt / num_samples
        gini_l = self.__gini(bad[0], cnt[0])
        gini_r = self.__gini(bad[1], cnt[1])
        gini = gini_l * ratio[0] + gini_r * ratio[1]
        return gini

    @ClassMethodAutoLog()
    def __gini(self, bad_samples, num_samples):
        """
        Gini formula.

        Args:
            bad_samples: float, bad nums.
            num_samples: int, num of samples.

        Return:
            gini value, float.
        """
        return 1 - (bad_samples / num_samples) ** 2 - \
                ((num_samples - bad_samples) / num_samples) ** 2

