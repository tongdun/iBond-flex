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

import math
import hashlib
from os import path

import numpy as np
from Crypto.Cipher import AES

from flex.crypto.key_exchange.api import make_agreement
from flex.tools.ionic import make_broadcast_channel
from flex.tools.base_algo import BaseProtocol


class Mixin:
    def _make_channel(self):
        self.var_chan = make_broadcast_channel(name="secure_alignment_broadcast",
                                               root=self.federal_info.coordinator,
                                               remote_group=self.federal_info.guest_host)

    def _encrypt(self, ids):
        seed = make_agreement(self.remote_id, self.key_size)
        seed_bytes = seed.to_bytes(math.ceil(seed.bit_length() / 8), 'big')
        aes = AES.new(hashlib.md5(seed_bytes).digest(), AES.MODE_ECB)
        encrypted_ids = list(map(lambda x: aes.encrypt(hashlib.md5(str(x).encode('utf-8')).digest()), ids))
        return encrypted_ids

    @staticmethod
    def _check_input(ids):
        # input
        if not isinstance(ids, list):
            raise TypeError(f"Input ids's type {type(ids)} is not list.")
        if len(np.array(ids).shape) != 1:
            raise ValueError("Input needs to be a 1-D list.")

    def _align(self, ids: list) -> list:
        # step1,step2
        self._check_input(ids)
        encrypted_ids = self._encrypt(ids)
        index = self.var_chan.allreduce(encrypted_ids, tag='secure_align_calc')
        # step4
        if self.federal_info.role == 'host':
            return [ids[i] for i in index[0]]
        else:
            return [ids[i] for i in index[1]]

    def _verify(self, ids: list) -> bool:
        # step4
        joined_ids = ''.join([str(x) for x in ids])
        hash_ids = hashlib.sha256(joined_ids.encode('utf-8')).digest()
        return self.var_chan.allreduce(hash_ids, tag='secure_align_verify')


class SALCoord(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict = None, algo_param: dict= None):
        """

        Args:
            federal_info:
            sec_param:
        """
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        super().__init__(federal_info, sec_param, algo_param)
        self._make_channel()

    #step3
    def align(self, *args, **kwargs) -> None:
        """
        Align the data from host and guest.
        """
        def intersection(data_pack):
            guest_ids, host_ids = data_pack
            _, host_idx, guest_idx = np.intersect1d(host_ids, guest_ids, return_indices=True)
            print('host_ids', host_ids)
            print('guest_ids', guest_ids)
            return (host_idx, guest_idx)

        self.var_chan.allreduce(intersection, tag='secure_align_calc')

    def verify(self, *args, **kwargs) -> None:
        """
        Check if samples are already aligned.
        """
        self.var_chan.allreduce(lambda x: x[0] == x[1], tag='secure_align_verify')


class SALGuest(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict, algo_param: dict = None):
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        super().__init__(federal_info, sec_param, algo_param)
        self.key_size = self.sec_param.key_exchange_size
        self.remote_id = self.federal_info.host[0]
        self._make_channel()

    def align(self, ids: list) -> list:
        return self._align(ids)

    def verify(self, ids: list) -> bool:
        return self._verify(ids)


class SALHost(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict, algo_param: dict = None):
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        super().__init__(federal_info, sec_param, algo_param)
        self.key_size = self.sec_param.key_exchange_size
        self.remote_id = self.federal_info.guest[0]
        self._make_channel()

    def align(self, ids: list) -> list:
        return self._align(ids)

    def verify(self, ids: list) -> bool:
        return self._verify(ids)
