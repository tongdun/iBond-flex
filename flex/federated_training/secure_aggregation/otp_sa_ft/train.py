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

import hashlib
from os import path
from typing import Union

import numpy as np
import torch

from flex.tools.base_algo import BaseProtocol
from flex.crypto.key_exchange.api import make_agreement
from flex.crypto.onetime_pad.api import generate_onetime_pad_encryptor
from flex.tools.ionic import make_broadcast_channel
from flex.tools.iterative_apply import iterative_divide


class Mixin:
    def _make_channel(self):
        self.broadcast_chan = make_broadcast_channel(name="otp_sa_ft_broadcast",
                                                     root=self.federal_info.coordinator,
                                                     remote_group=self.federal_info.guest_host)

    def _do_exchange(self, ciphertext) -> Union[list, np.ndarray, torch.Tensor]:
        self.broadcast_chan.gather(ciphertext, tag="theta")
        avg_theta = self.broadcast_chan.broadcast(tag="avg_theta")
        return avg_theta


class OTPSAFTCoord(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict = None):
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
        # step4
        theta = self.broadcast_chan.gather(tag="theta")
        avg_theta = iterative_divide(sum(theta).decode(), 2)
        self.broadcast_chan.broadcast(avg_theta, tag="avg_theta")


class OTPSAFTGuest(BaseProtocol, Mixin):
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

        self.seed = make_agreement(remote_id=self.federal_info.host[0], key_size=self.sec_param.key_exchange_size)
        # step1
        self.encryptor = generate_onetime_pad_encryptor(hashlib.sha512(str(self.seed).encode('utf-8')).digest())

    def exchange(self, theta: Union[list, np.ndarray, torch.Tensor]):
        enc_theta = self.encryptor.encrypt(theta, 1)
        # step5
        avg_theta = self._do_exchange(enc_theta)
        return avg_theta


class OTPSAFTHost(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict) -> Union[list, np.ndarray, torch.Tensor]:
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        super().__init__(federal_info, sec_param)
        self._make_channel()

        self.seed = make_agreement(remote_id=self.federal_info.guest[0], key_size=self.sec_param.key_exchange_size)
        # step1
        self.encryptor = generate_onetime_pad_encryptor(hashlib.sha512(str(self.seed).encode('utf-8')).digest())

    def exchange(self, theta: Union[list, np.ndarray, torch.Tensor]) -> Union[list, np.ndarray, torch.Tensor]:
        enc_theta = self.encryptor.encrypt(theta, -1)
        # step5
        avg_theta = self._do_exchange(enc_theta)
        return avg_theta
