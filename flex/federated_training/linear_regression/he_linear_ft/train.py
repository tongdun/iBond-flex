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

import numpy as np

from flex.constants import *
from flex.tools.base_algo import BaseProtocol
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
from flex.tools.ionic import make_broadcast_channel

class Mixin:
    def _make_channel(self):
        self.broadcast_chan = make_broadcast_channel(name="he_linear_ft_broadcast", root=self.federal_info.coordinator,
                                                     remote_group=self.federal_info.guest_host)

    def _do_exchange(self, x: np.ndarray):
        # step2 or step3
        ciphertext = self.encryptor.encrypt(x)
        self.broadcast_chan.gather(ciphertext, tag='ciphertext')

        # step4
        loss = self.broadcast_chan.broadcast(tag='loss')
        return loss


class HELinearFTCoord(BaseProtocol, Mixin):
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

        if self.sec_param.he_algo == CRYPTO_PAILLIER:
            self.encryptor, self.decryptor = generate_paillier_encryptor_decryptor(self.sec_param.he_key_length)
        else:
            raise NotImplementedError(f"Encryption algorithm {self.sec_param.he_algo} is not supported.")

        # step1
        self.broadcast_chan.broadcast(self.encryptor, tag='encryptor')

    def exchange(self):
        # step4
        ciphertexts = self.broadcast_chan.gather(tag='ciphertext')
        loss = self.decryptor.decrypt(sum(ciphertexts))
        self.broadcast_chan.broadcast(loss, tag='loss')
        return


class HELinearFTGuest(BaseProtocol, Mixin):
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

        # step1
        self.encryptor = self.broadcast_chan.broadcast(tag='encryptor')

    @staticmethod
    def _check_input(u, y):
        # input
        if not isinstance(u, np.ndarray):
            raise TypeError(f"Input u's type {type(u)} is not numpy.ndarray.")
        if len(u.shape) != 1:
            raise ValueError("Input u need to be a 1-D numpy.ndarray.")

        if not isinstance(y, np.ndarray):
            raise TypeError(f"Input y's type {type(y)} is not numpy.ndarray.")
        if len(y.shape) != 1:
            raise ValueError("Input y need to be a 1-D numpy.ndarray.")

    def exchange(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._check_input(u, y)
        x = u - y
        loss = self._do_exchange(x)
        return loss


class HELinearFTHost(BaseProtocol, Mixin):
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

        # step1
        self.encryptor = self.broadcast_chan.broadcast(tag='encryptor')

    @staticmethod
    def _check_input(u):
        if not isinstance(u, np.ndarray):
            raise TypeError(f"Input u's type {type(u)} is not numpy.ndarray.")
        if len(u.shape) != 1:
            raise ValueError("Input u need to be a 1-D numpy.ndarray.")

    def exchange(self, u: np.ndarray) -> np.ndarray:
        self._check_input(u)
        loss = self._do_exchange(u)
        return loss
