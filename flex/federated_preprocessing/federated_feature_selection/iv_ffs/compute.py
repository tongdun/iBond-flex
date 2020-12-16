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
import pandas as pd

from flex.tools.base_algo import BaseProtocol
from flex.tools.feature_iv_algo.feature_iv_algo import host_calc_bin_res, guest_iv_woe_value
from flex.tools.ionic import make_variable_channel
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
from flex.constants import CRYPTO_PAILLIER


class Mixin:
    def _make_channel(self):
        self.var_chan = make_variable_channel(name="iv_ffs_variable", endpoint1=self.federal_info.host[0],
                                              endpoint2=self.federal_info.guest[0])

    def _get_encryptor_decryptor(self):
        if self.sec_param.he_algo == CRYPTO_PAILLIER:
            self.encryptor, self.decryptor = generate_paillier_encryptor_decryptor(self.sec_param.he_key_length)
        else:
            raise NotImplementedError(f"Encryption algorithm {self.sec_param.he_algo} is not supported.")


class IVFFSHost(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict, algo_param: dict):
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        super().__init__(federal_info, sec_param, algo_param)
        self._make_channel()

    @staticmethod
    def _host_check(feature, is_continuous, split_info):
        if not isinstance(feature, pd.Series):
            raise TypeError(f"Input feature's type {type(feature)} is not pd.Series.")
        if not isinstance(is_continuous, bool):
            raise TypeError(f"Input feature's type {type(is_continuous)} is not bool.")
        if not isinstance(split_info, dict):
            raise TypeError(f"Input split_info's type {type(split_info)} is not dict.")

    def exchange(self, feature: pd.Series, is_continuous: bool, split_info: dict) -> float:
        self._host_check(feature, is_continuous, split_info)
        en_label = self.var_chan.recv(tag='encrypted_label')
        _, good_bad_nums = host_calc_bin_res(en_label, feature, split_info, is_continuous)
        self.var_chan.send(good_bad_nums, tag='encrypted_good_bad_nums')
        iv_value = self.var_chan.recv(tag='iv_value')
        return iv_value


class IVFFSGuest(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict, algo_param: dict):
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        super().__init__(federal_info, sec_param, algo_param)
        self._make_channel()

        if self.sec_param.he_algo == CRYPTO_PAILLIER:
            self.encryptor, self.decryptor = generate_paillier_encryptor_decryptor(self.sec_param.he_key_length)
        else:
            raise NotImplementedError(f"Encryption algorithm {self.sec_param.he_algo} is not supported.")

    @staticmethod
    def _guest_check(label):
        if not isinstance(label, (np.ndarray, pd.Series)):
            raise TypeError(f"Input label's type {type(label)} is not pd.Series or np.ndarray.")

    def exchange(self, label: Union[np.ndarray, pd.Series]) -> None:
        self._guest_check(label)
        if isinstance(label, pd.Series):
            label = label.values
        en_label = self.encryptor.encrypt(label)
        self.var_chan.send(en_label, tag='encrypted_label')
        good_bad_nums = self.var_chan.recv(tag='encrypted_good_bad_nums')
        _, iv_value = guest_iv_woe_value(label, good_bad_nums, self.decryptor, self.algo_param.adjust_value)
        self.var_chan.send(iv_value, tag='iv_value')
