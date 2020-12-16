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

import secrets
from os import path
from typing import Tuple

import numpy as np

from flex.constants import CRYPTO_PAILLIER
from flex.tools.base_algo import BaseProtocol
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor, generate_paillier_encryptor
from flex.tools.ionic import make_variable_channel


class Mixin:
    """Common for host, guest and coordinator
    """
    def _make_channel(self):
        self.var_chan = make_variable_channel('he_otp_lr_ft1_variable_H2G', self.federal_info.host[0],
                                                     self.federal_info.guest[0])

    def _get_encryptor_decryptor(self):
        """Get encryptor and decryptor according to sec_parm
        """
        if self.sec_param.he_algo == CRYPTO_PAILLIER:
            self.encryptor, self.decryptor = generate_paillier_encryptor_decryptor(self.sec_param.he_key_length)
        else:
            raise NotImplementedError(f'Encryption algorithm {self.sec_param.he_algo} is not supported.')

    def _get_encryptor(self, pub_key):
        """Get encryptor based on pub_key
        """
        if self.sec_param.he_algo == CRYPTO_PAILLIER:
            self.encryptor = generate_paillier_encryptor(pub_key)
        else:
            raise NotImplementedError(f'Encryption algorithm {self.sec_param.he_algo} is not supported.')


class HEOTPLR1Guest(BaseProtocol, Mixin):
    """HE_OTP_LR_FT1 Guest side
    """

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
        self._get_encryptor_decryptor()

        # Step 1
        self.var_chan.send(self.encryptor, tag='encryptor')

    @staticmethod
    def _check_input(theta, features, labels):
        if not isinstance(theta, np.ndarray):
            raise TypeError('Input theta is not np.ndarray.')
        if not isinstance(features, np.ndarray):
            raise TypeError('Input features is not np.ndarray.')
        if not isinstance(labels, np.ndarray):
            raise TypeError('Input labels is not np.ndarray.')
        if len(theta.shape) != 1:
            raise ValueError('Input theta need to be a 1-D np.ndarray of [Num_features].')
        if len(features.shape) != 2:
            raise ValueError('Input features need to be a 2-D np.ndarray of [Batchsize, Num_features].')
        if len(labels.shape) != 1:
            raise ValueError('Input labels need to be a 1-D np.ndarray of [Num_features].')
        if features.shape[0] != labels.shape[0]:
            raise ValueError('Batchsize not match')
        if theta.shape[0] != features.shape[1]:
            raise ValueError('Number of features not match')

    def exchange(self, theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Iteration part
        """
        self._check_input(theta, features, labels)
        # Step 3
        u2 = self.var_chan.recv(tag='u2')
        u1 = theta.dot(features.T)
        u = u1 + u2
        h_x = 1 / (1 + np.exp(-u))
        enc_diff_y = self.encryptor.encrypt(labels - h_x)
        self.var_chan.send(enc_diff_y, tag='enc_diff_y')

        # Step 5
        enc_padded_grads = self.var_chan.recv(tag='enc_padded_grads')
        padded_grads = self.decryptor.decrypt(enc_padded_grads)
        self.var_chan.send(padded_grads, tag='padded_grads')

        batch_size = features.shape[0]
        # print('labels', labels)
        # print('h_x', h_x)
        # print('u', u)
        # print('features', features)
        # print('----')
        grads = (-1 / batch_size) * ((labels - h_x).dot(features))
        return h_x, grads


class HEOTPLR1Host(BaseProtocol, Mixin):
    """HE_OTP_LR_FT1 Host side
    """

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

        # Step 1
        self.encryptor = self.var_chan.recv(tag='encryptor')

    @staticmethod
    def _check_input(theta, features):
        if not isinstance(theta, np.ndarray):
            raise TypeError('Input theta is not np.ndarray.')
        if not isinstance(features, np.ndarray):
            raise TypeError('Input features is not np.ndarray.')
        if len(theta.shape) != 1:
            raise ValueError('Input theta need to be a 1-D np.ndarray of [Num_features].')
        if len(features.shape) != 2:
            raise ValueError('Input features need to be a 2-D np.ndarray of [Batchsize, Num_features].')
        if theta.shape[0] != features.shape[1]:
            raise ValueError('Number of features not match')

    def exchange(self, theta: np.ndarray, features: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Iteration part
        """
        self._check_input(theta, features)
        # Step 2
        u2 = theta.dot(features.T)
        self.var_chan.send(u2, tag='u2')
        # Step 4
        enc_diff_y = self.var_chan.recv(tag='enc_diff_y')
        batch_size = features.shape[0]
        enc_grads = (-1 / batch_size) * (enc_diff_y.dot(features))

        rand = secrets.SystemRandom()
        r = np.array([rand.randint(-2 ** 64, 2 ** 64) / 2 ** 48 for i in range(len(enc_grads))])
        enc_r = self.encryptor.encrypt(r)
        enc_padded_grads = enc_grads + enc_r
        self.var_chan.send(enc_padded_grads, tag='enc_padded_grads')
        # Step 6
        padded_grads = self.var_chan.recv(tag='padded_grads')
        grads = padded_grads - r
        return grads
