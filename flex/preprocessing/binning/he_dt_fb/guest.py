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

from .common import DTBaseModel
from flex.cores.base_model import send_pubkey_signal
from flex.utils import ClassMethodAutoLog


class HEDTFBGuest(DTBaseModel):
    """
    Federated data preprocessing: Decision Tree binning in guest.
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
        # share public key
        send_pubkey_signal(self.key_channel, self.pf_ecc.en)

    @ClassMethodAutoLog()
    def pre_exchange(self, label: np.ndarray, *args, **kwargs) -> None:
        """
        First protocol of DT binning protocol, send encrypt label mess to host
        Arg:
            label: samples label

        Return:
             None
        ----

        **Example:**
        >>> label = np.array([0, 1, 0, 1, 0, 0])
        >>> self.pre_exchange(label)
        """
        # encrypt label
        label = self.pf_ecc.encrypt(label)

        # send label to host
        self.label_channel.send(label)

    @ClassMethodAutoLog()
    def exchange(self, label: np.ndarray, *args, **kwargs) -> None:
        """
        This is a guest exchange function

        Args:
            label: shape (n,1) numpy array or pandas series. Input label to help host calculate DT binning.

        Returns:
            None
        ----

        **Example:**
        >>> protocol = HEDTFBGuest(federal_info, sec_param, algo_param)
        >>> label = np.array([0,0,0,0,0,0,0,1,1,1,1,1,0,0,1])
        >>> protocol.exchange(label=label)
        """
        # get unique values msg from host
        judge_type = self.judge_channel.recv()
        if judge_type:
            return

        # data type check
        self._check_input(label)
        self.logger.info("Guest complete input label's check")

        # params node_num
        self._gen_node_num(label.shape[0])

        # step3: guest help host calculate split points
        on_set = True
        while on_set:
            # judge tree split is continue
            on_set = self.judge_tree.recv()
            if not on_set:
                break

            bin_info_dict = self.bin_info_channel.recv()
            self._calc_best_split(bin_info_dict)
        self.logger.info('Guest complete this term DT bin protocol')

    @ClassMethodAutoLog()
    def _calc_best_split(self, bin_info_dict: Dict):
        """
        This method mainly calc best split point

        Args:
            bin_info_dict: dict, storage left/right node sample/label message
        """
        gain = -np.inf
        split = None
        for index in bin_info_dict:
            # bin msg is None
            if len(bin_info_dict[index]) == 0:
                break
            # calc best split msg
            value = bin_info_dict[index]
            count = value[0], value[1]
            y = self.pf_ecc.decrypt(value[2]), self.pf_ecc.decrypt(value[3])
            bin_gain = self._gini_gain(count, y)

            if bin_gain >= gain:
                split = index
        self.split_channel.send(split)

    @staticmethod
    @ClassMethodAutoLog()
    def _gini_gain(count: tuple, y: tuple) -> float:
        """
        Calculation of gain by gini
        """
        # gain of parent node
        num = count[0] + count[1]
        num_p = y[0] + y[1]
        num_n = num - num_p
        gini_o = 1 - (num_p / num) ** 2 - (num_n / num) ** 2

        # gain of left child
        num_nl = count[0] - y[0]
        gini_l = 1 - (y[0] / count[0]) ** 2 - (num_nl / count[0]) ** 2

        # gain of right child
        num_nr = count[1] - y[1]
        gini_r = 1 - (y[1] / count[1]) ** 2 - (num_nr / count[1]) ** 2

        # left/right split gain
        gini_lr = (count[0] / num) * gini_l + (count[1] / num) * gini_r

        # final gain
        bin_gain = gini_o - gini_lr

        return bin_gain
