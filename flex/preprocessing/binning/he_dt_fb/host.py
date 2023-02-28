#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np

from .common import DTBaseModel
from flex.utils import ClassMethodAutoLog
from flex.cores.base_model import get_pubkey_signal
from flex.algo_config import HE_DT_FB


class HEDTFBHost(DTBaseModel):
    """
        Federated data preprocessing: Decision Tree binning in host.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        DTBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)
        # get pubkey
        self.pf_ecc.en = get_pubkey_signal(self.key_channel)

    @ClassMethodAutoLog()
    def pre_exchange(self, *args, **kwargs) -> np.ndarray:
        """
        First protocol of DT binning protocol, get encrypt label msg from guest

        Returns:
             encrypt message of label
        ----

        **Example:**
        >>> self.pre_exchange()
        """
        # step2: host receive the encrypted label.
        label_en = self.label_channel.recv()

        return label_en

    @ClassMethodAutoLog()
    def exchange(self, en_label: np.ndarray,
                 feature: Optional[np.ndarray],
                 *args, **kwargs) -> np.ndarray:
        """
        This is a host exchange function

        Args:
            en_label: encrypt label
            feature: shape (n,1) numpy array or pandas series. the feature to get split points.

        Returns:
            dict/None, the result is split points by decision tree binning.
        ----

        **Example:**
        >>> en_label = np.array([0,1,0,0,1,1,1])
        >>> feature = np.array([39,50,38,53,28,37,49,52,31,42,37,30,23,32,40,34])
        >>> split_points = self.exchange(en_label, feature)
        """
        # unique value is less than threshold, send msg to guest
        if len(np.unique(feature)) <= HE_DT_FB['category_threshold']:
            thr = HE_DT_FB['category_threshold']
            warnings.warn(f'unique value of this feature is no more than {thr}')

            self.judge_channel.send(True)
            return np.unique(feature)
        else:
            self.judge_channel.send(False)

        # data type check
        self._check_input(feature)
        self.logger.info("Host complete input feature's check")

        # params node_num
        self._gen_node_num(feature.shape[0])

        #  step4: host calculate the candidate split points, send to guest and get the best split points.
        self.init_points = self._init_split(feature)
        self._dt_split_points(feature, en_label)
        self.logger.info("Host has calculated the best split points.")

        self.judge_tree.send(False)
        self.logger.info("Host has sent the symbol of finish DT binning.")
        return self.split_points

    @ClassMethodAutoLog()
    def _dt_split_points(self,
                         sample_set: np.ndarray,
                         en_label: np.ndarray) -> None:
        """
        THis method mainly construct tree model, generate split points

        Args:
            sample_set: datasets
            en_label: encrypt label msg

        Returns:
             None
        """
        # num of split points more then max_bin_num
        if len(self.split_points) >= self.algo_param.max_bin_num:
            return

        # notifies guest to continue the loop
        self.judge_tree.send(True)

        # data split according inits split points
        bin_info = dict()
        for i, point in enumerate(self.init_points):
            result = _get_bin_index(sample_set, en_label, point)
            if result is not None:
                bin_info[point] = result
        self.bin_info_channel.send(bin_info)

        # get split result from guest
        split = self.split_channel.recv()

        if split is not None:
            # split points update
            self.split_points = np.append(split, self.split_points)
            self._update_split_points()

            # split data to left/right node
            left_idx, right_idx = _calc_lr_index(sample_set, split)
            sample_set_left = sample_set[left_idx]
            sample_set_right = sample_set[right_idx]
            en_label_left = en_label[left_idx]
            en_label_right = en_label[right_idx]

            # continue update split points
            if sample_set_left.shape[0] >= self.algo_param.node_num * 2:
                self._dt_split_points(sample_set_left, en_label_left)
            if sample_set_right.shape[0] >= self.algo_param.node_num * 2:
                self._dt_split_points(sample_set_right, en_label_right)

    @ClassMethodAutoLog()
    def _update_split_points(self) -> None:
        self.split_points = np.unique(self.split_points)


@ClassMethodAutoLog()
def _get_bin_index(data: np.ndarray, en_label: np.ndarray,
                   node: float) -> Optional[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    This method mainly split data and update encrypt label msg

    Args:
        data: np.array, origin datasets
        en_label: np.array, encrypt label msg
        node: float, split points

    Returns:
        tuple, left/right num of samples, left/right sum of encrypt label
    """
    left_idx, right_idx = _calc_lr_index(data, node)

    if len(right_idx) == 0 or len(left_idx) == 0:
        return None

    right_label = np.array(en_label)[right_idx]
    left_label = np.array(en_label)[left_idx]

    return len(left_label), len(right_label), sum(left_label), sum(right_label)


@ClassMethodAutoLog()
def _calc_lr_index(data: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method mainly calc left/right index for data less/more than threshold

    Args:
        data: datasets
        threshold: left/right node split threshold

    Returns:
         left/right node index
    ----

    **Example:**
    >>> data = np.array([2,6,1,7,3,9,3,2])
    >>> threshold = 3.9
    >>> _calc_lr_index(data, threshold)
    """
    left_index = np.where(data <= threshold)[0]
    right_index = np.where(data > threshold)[0]

    return left_index, right_index
