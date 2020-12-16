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

from os import path
from typing import Union

import numpy as np
import torch

from flex.tools.base_algo import BaseProtocol
from flex.crypto.key_exchange.api import make_agreement
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
from flex.tools.ionic import make_broadcast_channel
from flex.crypto.onetime_pad.iterative_add import iterative_add
from flex.tools.iterative_apply import iterative_encryption, iterative_decryption, iterative_divide


class Mixin:
    def _make_channel(self):
        self.broadcast_chan = make_broadcast_channel(name="he_sa_ft_broadcast",
                                                     root=self.federal_info.coordinator,
                                                     remote_group=self.federal_info.guest_host)

    def _do_exchange(self, theta: Union[list, np.ndarray, torch.Tensor], remote_id: str) -> Union[list, np.ndarray, torch.Tensor]:
        # step1
        seed = make_agreement(remote_id, key_size=self.sec_param.key_exchange_size)
        encryptor, decryptor = generate_paillier_encryptor_decryptor(self.sec_param.he_key_length, seed)
        # step2
        enc_theta = iterative_encryption(encryptor, theta)
        self.broadcast_chan.gather(enc_theta, tag="theta")
        # step4
        enc_sum_theta = self.broadcast_chan.broadcast(tag="sum_theta")
        sum_theta = iterative_decryption(decryptor, enc_sum_theta)
        avg_theta = iterative_divide(sum_theta, 2.0)
        return avg_theta


class HESAFTCoord(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict):
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        super().__init__(federal_info, sec_param)
        self._make_channel()

    def exchange(self, *args, **kwargs):
        # step3
        enc_theta_list = self.broadcast_chan.gather(tag="theta")
        sum_enc_theta = enc_theta_list[0]
        for i in range(1, len(enc_theta_list)):
            sum_enc_theta = iterative_add(sum_enc_theta, enc_theta_list[i])
        self.broadcast_chan.broadcast(sum_enc_theta, tag="sum_theta")
        return


class HESAFTGuest(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict):
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        super().__init__(federal_info, sec_param)
        self._make_channel()

    def exchange(self, theta: Union[list, np.ndarray, torch.Tensor]) -> Union[list, np.ndarray, torch.Tensor]:
        avg_theta = self._do_exchange(theta, remote_id=self.federal_info.host[0])
        return avg_theta


class HESAFTHost(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict):
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        super().__init__(federal_info, sec_param)
        self._make_channel()

    def exchange(self, theta: Union[list, np.ndarray, torch.Tensor]) -> Union[list, np.ndarray, torch.Tensor]:
        avg_theta = self._do_exchange(theta, remote_id=self.federal_info.guest[0])
        return avg_theta
