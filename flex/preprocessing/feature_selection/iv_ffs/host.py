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

from typing import Union, Dict, List, Optional, Tuple, Any

import numpy as np

from flex.algo_mixin.bin.histogram import HeteroBin
from .common import IVBaseModel
from flex.utils import ClassMethodAutoLog


class IVFFSHost(IVBaseModel):
    """
        Federated data preprocessing: calculate IV value.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        IVBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

    @ClassMethodAutoLog()
    def pre_exchange(self,
                     tag='*',
                     *args, **kwargs) -> np.ndarray:
        """
        First protocol of IV value protocol, get encrypt label mess from guest

        Arg:
            tag: str, identification of communication.

        Returns:
             encrypt message of label
        ----

        **Example**
        >>> iv = IVFFSHost(federal_info, sec_param, algo_param)
        >>> iv.pre_exchange()
        """
        # step2: host receives encrypted label from guest
        en_labels = self.label_channel.broadcast(tag=tag)
        self.logger.info("host has received encrypted labels from guest.")

        return en_labels

    @ClassMethodAutoLog()
    def exchange(self, feature: Optional[np.ndarray] = None,
                 en_labels: Optional[np.ndarray] = None,
                 is_category: Optional[bool] = False,
                 data_null: Optional[bool] = False,
                 split_list: Optional[List] = None,
                 tag: str = '*',
                 *args, **kwargs) -> Union[Tuple[Any, np.ndarray, Any], Tuple[None, None, None]]:
        """
        This is a host exchange function

        Args:
            feature: shape (n,1) numpy array or pandas series. the feature to get split points.
            en_labels: encrypted labels receive from guest.
            is_category: bool, feature's type is category or not, decides the result of return.
            data_null: bool, feature has missing values or not.
            split_list: list, split point value of feature.
            tag: str, identification of communication.

        Returns:
            float, bool: the first result is ks value,
            the second result is ks value is greater or equal to threshold or not.
        ----

        **Example**
        >>> protocol = IVFFSHost(federal_info, sec_param, algo_param)
        >>> feature = np.array([39,50,38,53,28,37,49,52,31,42,37,30,23,32,40,34])
        >>> en_labels = protocol.pre_exchange()
        >>> is_category = False
        >>> data_null = False
        >>> split_info = [30,  36, 43,  50]
        >>> ks_val, ks_bool = protocol.exchange(feature, en_labels, is_category, data_null, split_list)
        """
        if feature is None:
            self.good_bad_nums_channel.gather(None, tag=tag)
            self.iv_woe_value_channel.scatter(tag=tag)
            return None, None, None
        else:
            self._check_host_input(feature, is_category, data_null)
            self.logger.info("host complete input feature and feature type's check.")

            # step2: host calculates the encrypted bad_good_nums, sends the result to guest.
            bad_total_nums = self._host_calc_bin_res(en_labels, feature, split_list, is_category, data_null)
            self.logger.info(f"host has calculated bad and total nums.")

            self.good_bad_nums_channel.gather(bad_total_nums, tag=tag)
            self.logger.info("host completes send good and bad nums to guest.")

            iv_woe_value = self.iv_woe_value_channel.scatter(tag=tag)
            self.logger.info("host has received iv and woe values from guest.")

            iv_value = np.sum(iv_woe_value[0])
            woe_value = iv_woe_value[1]

            return woe_value, iv_value, iv_value >= self.algo_param.iv_thres

    @ClassMethodAutoLog()
    def _host_calc_bin_res(self,
                           en_label: np.ndarray,
                           feature: np.ndarray,
                           split_list: List,
                           is_category: bool,
                           data_null: bool) -> List:
        """
            host role calculates the bin results.
        """
        self.bin_obj = HeteroBin()
        bad_total_nums = []
        if is_category:
            self.bin_obj.get_discrete_bin(feature)
        else:
            self.bin_obj.get_bin_hist(feature, split_list)
        if data_null:
            self.bin_obj.get_none_type_bin(feature)
        good_num, bad_num, total_num = self.bin_obj.en_good_bad_calc(en_label)
        bad_total_nums.append(bad_num)
        bad_total_nums.append(total_num)
        return bad_total_nums
