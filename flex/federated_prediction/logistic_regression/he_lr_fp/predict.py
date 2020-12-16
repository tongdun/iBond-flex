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
from flex.tools.ionic import make_variable_channel


class HELRFPCoord(BaseProtocol):
    def __init__(self, federal_info: dict, sec_param: dict = None):
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        super().__init__(federal_info, sec_param)
        self.variable_chan_guest_coordinator = make_variable_channel('he_lr_fp_variable_gc', self.federal_info.guest[0],
                                                                     self.federal_info.coordinator)
        self.variable_chan_host_coordinator = make_variable_channel('he_lr_fp_variable_hc', self.federal_info.host[0],
                                                                    self.federal_info.coordinator)

        # step1,get pk, get [u1]
        self.encryptor = self.variable_chan_guest_coordinator.recv(tag='pub_key')

    def exchange(self, *args, **kwargs):
        # step1： C receiver u1
        enc_u1 = self.variable_chan_guest_coordinator.recv(tag='enc_u1')

        # step2: send pk to P2
        self.variable_chan_host_coordinator.send(self.encryptor, tag='pub_key')

        # step4:
        enc_u2 = self.variable_chan_host_coordinator.recv(tag='enc_u2')
        self.variable_chan_guest_coordinator.send(enc_u1 + enc_u2, tag='enc_u')
        return


class HELRFPGuest(BaseProtocol):
    def __init__(self, federal_info: dict, sec_param: dict):
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        super().__init__(federal_info, sec_param)
        self.variable_chan_guest_coordinator = make_variable_channel('he_lr_fp_variable_gc', self.federal_info.guest[0],
                                                                     self.federal_info.coordinator)

        if self.sec_param.he_algo == CRYPTO_PAILLIER:
            self.encryptor, self.decryptor = generate_paillier_encryptor_decryptor(self.sec_param.he_key_length)
        else:
            raise NotImplementedError(f"Encryption algorithm {self.sec_param.he_algo} is not supported.")

        self.variable_chan_guest_coordinator.send(self.encryptor, tag='pub_key')

    @staticmethod
    def _check_input(u1):
        if not isinstance(u1, np.ndarray):
            raise TypeError(f"Input u1's type {type(u1)} is not numpy.ndarray.")
        if len(u1.shape) != 1:
            raise ValueError("Input u1 need to be a 1-D numpy.ndarray.")

    def exchange(self, u1: np.ndarray):
        self._check_input(u1)
        # step1: send [u1] to C
        enc_u1 = self.encryptor.encrypt(u1)
        self.variable_chan_guest_coordinator.send(enc_u1, tag='enc_u1')

        # step4:
        enc_u = self.variable_chan_guest_coordinator.recv(tag='enc_u')
        u = self.decryptor.decrypt(enc_u)
        return u


class HELRFPHost(BaseProtocol):
    def __init__(self, federal_info: dict, sec_param: dict):
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        super().__init__(federal_info, sec_param)
        self.variable_chan_host_coordinator = make_variable_channel('he_lr_fp_variable_hc', self.federal_info.coordinator,
                                                                    self.federal_info.host[0])

        # step3, receive pk
        self.encryptor = self.variable_chan_host_coordinator.recv(tag='pub_key')

    @staticmethod
    def _check_input(u2):
        # input
        if not isinstance(u2, np.ndarray):
            raise TypeError(f"Input u2's type {type(u2)} is not numpy.ndarray.")
        if len(u2.shape) != 1:
            raise ValueError("Input u2 need to be a 1-D numpy.ndarray.")

    def exchange(self, u2: np.ndarray):
        self._check_input(u2)
        # step3: send u2 to C
        enc_u2 = self.encryptor.encrypt(u2)
        self.variable_chan_host_coordinator.send(enc_u2, tag='enc_u2')
        return
