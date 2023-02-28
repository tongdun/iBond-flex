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
from flex.algo_mixin.bin.histogram import HeteroBin
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
                     tag: str = '*',
                     *args, **kwargs) -> None:
        """
        First protocol of hetero tree split protocol, send encrypt label mess to host
        Arg:
            label: samples label.
            tag: str, identification of communication.

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
        self.label_channel.broadcast(en_labels, tag=tag)
        self.logger.info("guest complete label's encryption and send to host")

    @ClassMethodAutoLog()
    def exchange(self,
                 label: np.ndarray,
                 tag: str = '*',
                 *args, **kwargs) -> None:
        """
        This is a guest exchange function

        Args:
            label: shape (n,1) numpy array. Input label to help host calculate KS value.
            tag: str, identification of communication.

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

        bad_total_nums = self.good_bad_nums_channel.gather(tag=tag)
        self.logger.info("guest receives bad_total_nums from host")

        iv_woe_list = []
        for bad_total_num in bad_total_nums:
            if bad_total_num is None:
                iv_woe_list.append([None, None])
            else:
                woe_value, iv_value = bin_obj.calc_woe_iv(y=label,
                                                          total_num=bad_total_num[1],
                                                          en_bad_num=bad_total_num[0],
                                                          decrypter=self.pf_ecc,
                                                          adjust_value=self.algo_param.adjust_value)
                iv_woe_list.append([iv_value, woe_value])
        self.logger.info("guest completes iv and woe value's calculation")

        self.iv_woe_value_channel.scatter(iv_woe_list, tag=tag)
        self.logger.info(f"guest sends iv and woe value's to host")
