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

from typing import Dict, List, Optional

import numpy as np

from .common import IVBaseModel
from flex.preprocessing.feature_selection.iv_ffs.mixin import HeteroBin
from flex.utils import ClassMethodAutoLog


class IVFFSGuest(IVBaseModel):
    """
        Federated data preprocessing: calculate KS value.
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
                     label: np.ndarray,
                     *args, **kwargs) -> None:
        """
        First protocol of hetero tree split protocol, send encrypt label mess to host
        Arg:
            label: samples label

        Return:
             None
        ----

        **Example**:
        >>> label = np.array([0, 1, 0, 1, 0, 0])
        >>> IVFFSGuest.pre_exchange(label)
        """
        self._check_guest_input(label)
        self.logger.info("guest complete input label's check")

        # encrypt label
        en_labels = self.pf_ecc.encrypt(label)

        # send label to host
        self.label_channel.broadcast(en_labels)
        self.logger.info("guest complete label's encryption and send to host")

    @ClassMethodAutoLog()
    def exchange(self,
                 label: np.ndarray,
                 *args, **kwargs) -> None:
        """
        This is a guest exchange function

        ----
        Args:
            label: shape (n,1) numpy array. Input label to help host calculate KS value.

        Returns:
            None
        ----

        **Examples**
        >>> protocol = IVFFSGuest(federal_info, sec_param, algo_param)
        >>> label = np.array([0,0,0,0,0,0,0,1,1,1,1,1,0,0,1])
        >>> protocol.exchange(label=label)
        """
        # step1: guest calculates labels info and encrypts the label, then sends to host.
        self._check_guest_input(label)
        bin_obj = HeteroBin()

        good_bad_nums = self.good_bad_nums_channel.gather()
        self.logger.info("guest receives good_bad_nums from host")

        iv_woe_list = []
        for good_bad_num in good_bad_nums:
            if good_bad_num is None:
                iv_woe_list.append([None, None])
            else:
                woe_value, iv_value = bin_obj.calc_woe_iv(y=label,
                                                          en_good_num=good_bad_num[0],
                                                          en_bad_num=good_bad_num[1],
                                                          decryptor=self.pf_ecc,
                                                          adjust_value=self.algo_param.adjust_value)
                iv_woe_list.append([iv_value, woe_value])
        self.logger.info("guest completes iv and woe value's calculation")

        self.iv_woe_value_channel.scatter(iv_woe_list)
        self.logger.info("guest sends iv and woe value's to host")
