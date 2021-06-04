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

from typing import Union, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .mixin import HeteroBin
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
    def pre_exchange(self, *args, **kwargs) -> np.ndarray:
        """
        First protocol of IV value protocol, get encrypt label mess from guest

        Arg:
            None

        Returns:
             encrypt message of label
        ----

        **Example**
        >>> iv = IVFFSHost(federal_info, sec_param, algo_param)
        >>> iv.pre_exchange()
        """
        # step2: host receives encrypted label from guest
        en_labels = self.label_channel.broadcast()
        self.logger.info("host has received encrypted labels from guest.")

        return en_labels

    @ClassMethodAutoLog()
    def exchange(self,
                 feature: Optional[Union[pd.Series, np.ndarray]] = None,
                 en_labels: Optional[np.ndarray] = None,
                 is_category: Optional[bool] = None,
                 data_null: Optional[bool] = None,
                 split_info: Optional[Dict] = None,
                 *args, **kwargs) -> Union[Tuple[float, bool], Tuple[None, None]]:
        """
        This is a host exchange function

        Args:
            feature: shape (n,1) numpy array or pandas series. the feature to get split points.
            en_labels: encrypted labels receive from guest.
            is_category: bool, feature's type is category or not, decides the result of return.
            data_null: bool, feature has missing values or not.
            split_info: dict, the split info from binning.

        Returns:
            float, bool: the first result is ks value and the second result is ks value is greater or equal to threshold or not.
        ----

        **Example**
        >>> protocol = IVFFSHost(federal_info, sec_param, algo_param)
        >>> feature = pd.Series([39,50,38,53,28,37,49,52,31,42,37,30,23,32,40,34])
        >>> en_labels = protocol.pre_exchange()
        >>> is_category = False
        >>> data_null = False
        >>> split_info = {'split_points': np.array([30,  36, 43,  50])}
        >>> ks_val, ks_bool = protocol.exchange(feature, en_labels, is_category, data_null, split_info)
        """
        if feature is None:
            self.good_bad_nums_channel.gather(None)
            self.iv_woe_value_channel.scatter()
            return None, None
        else:
            self._check_host_input(feature, is_category, data_null)
            self.logger.info("host complete input feature and feature type's check.")

            if isinstance(feature, np.ndarray):
                feature = pd.Series(feature)

            # step2: host calculates the encrypted bad_good_nums, sends the result to guest.
            good_bad_nums = self._host_calc_bin_res(en_labels, feature, split_info, is_category, data_null)
            self.logger.info("host has calculated good and bad nums.")

            self.good_bad_nums_channel.gather(good_bad_nums)
            self.logger.info("host completes send good and bad nums to guest.")

            iv_woe_value = self.iv_woe_value_channel.scatter()
            self.logger.info("host has received iv and woe values from guest.")

            iv_value = iv_woe_value[0]
            woe_value = iv_woe_value[1]
            self.bin_obj.bin_result['iv'] = [np.sum(iv_value), iv_value[-1]] if data_null else [np.sum(iv_value)]
            self.bin_obj.bin_result['woe'] = woe_value
            bin_result_save = self.bin_obj.bin_result
            for s in ['data', 'index']:
                bin_result_save.pop(s)
            self.logger.info("host has calculated the bin results.")

            return bin_result_save, bin_result_save['iv'][0] >= self.algo_param.iv_thres

    @ClassMethodAutoLog()
    def _host_calc_bin_res(self,
                           en_label: np.ndarray,
                           feature: pd.Series,
                           split_info: Dict,
                           is_category: bool,
                           data_null: bool) -> List:
        """
            host role calculates the bin results.
        """
        self.bin_obj = HeteroBin()
        good_bad_nums = []
        if is_category:
            self.bin_obj.get_discrete_bin(feature)
        else:
            self.bin_obj.get_bin_hist(feature, split_info)
        if data_null:
            self.bin_obj.get_nonetype_bin(feature)
        good_num, bad_num = self.bin_obj.en_good_bad_calc(en_label)
        good_bad_nums.append(good_num)
        good_bad_nums.append(bad_num)
        return good_bad_nums
