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

import numpy as np
from typing import List, Union, Dict, Optional
import secrets
import math
import hashlib

from Crypto.Cipher import AES
try:
    from gmcrypto import sm4
except ImportError:
    sm4 = None
    print('Warning, gmcrypto is not installed, SM4 will not be available.')


from flex.constants import CRYPTO_PAILLIER, CRYPTO_AES, \
    CRYPTO_ONETIME_PAD, CRYPTO_SECRET_SHARING, CRYPTO_SM4
from flex.constants import CRYPTO_KEY_LENGTH, CRYPTO_NK, CRYPTO_NONE_PARAM, CRYPTO_PRECISION
from flex.utils import ClassMethodAutoLog
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
from flex.crypto.onetime_pad.api import generate_onetime_pad_encryptor


class EncryptModel:
    """
    mainly given encrypt/decrypt mess from different encrypt method
    """
    def __init__(self, method: Optional[str] = None,
                 param: Optional[Dict] = None,
                 seed: Optional[Union[int, List]] = None):
        """
        init contain param inits and secure method inits
        ----

        Args:
            method: str, security method; None means no encrypted method
            param: Dict, params for security method
            seed: int/list, strorage diffie hellman seed value
        ----

        **Example**:
        >>> method = "paillier"
        >>> param = {"key_length": 1024}
        >>> seed = None
        >>> EncryptModel(method, param, None)
        """
        # seed value
        self.seed = seed

        # encrypt method is None
        self.empty = method is None

        # encrypt param inits
        self._init_encrypt_param(method=method, param=param)

        # init encrypt
        encrypt_method = {
            CRYPTO_PAILLIER: self._generate_paillier,
            CRYPTO_ONETIME_PAD: self._generate_onetime_pad,
            CRYPTO_AES: self._generate_aes,
            CRYPTO_SM4: self._generate_sm4
        }
        if not self.empty:
            if self.method not in [CRYPTO_SECRET_SHARING]:
                self.en, self.de = encrypt_method[self.method]()

    def _init_encrypt_param(self, method: Optional[str] = None,
                            param: Optional[Dict] = None) -> None:
        """
        This method mainly inits the encrypt parameters
        ----

        Args:
            method: str, security method; None means no encrypted method
            param: Dict, params for security method
        """
        if not self.empty:
            self.method = method

            # paillier, sm4, aes, onetime_pad, key_echange and ecc, sec_param only key_length
            if method in CRYPTO_KEY_LENGTH:
                self.key_length = param['key_length']

            # md5, sm3, no required of encrypt params
            elif method in CRYPTO_NONE_PARAM:
                pass

            # secret sharing, has parameter of precision
            elif method in CRYPTO_PRECISION:
                self.precision = param['precision']

            # OT's params
            elif method in CRYPTO_NK:
                self.n = param['n']
                self.k = param['k']

    def _generate_paillier(self):
        """
        Generate paillier encryptor and decrypter
        """
        pail_en, pail_de = generate_paillier_encryptor_decryptor(self.key_length)

        return pail_en, pail_de

    def _generate_aes(self):
        """
        Generate ase encryptor/decrypter
        """
        # check seed value
        self._check_seed_msg()

        # change seed value to bytes
        seed_bytes = self.seed.to_bytes(math.ceil(self.seed.bit_length() / 8), 'big')

        # generate ase encryptor
        encryptor = AES.new(seed_bytes[:math.ceil(self.key_length / 8)], AES.MODE_ECB)

        return encryptor, encryptor

    def _generate_onetime_pad(self):
        """
        Generate one_time_pad encryptor/decrypter
        """
        # check seed value
        self._check_seed_msg()

        # change seed value to bytes
        seed_bytes = self.seed.to_bytes(math.ceil(self.seed.bit_length() / 8), 'big')

        # generate onetime_pad encryptor
        en = generate_onetime_pad_encryptor(seed_bytes[:math.ceil(self.key_length / 8)])

        return en, en

    def _generate_sm4(self):
        """
        Generate ase encryptor/decrypter
        """
        if sm4 is None:
            raise RuntimeError("SM4 is not supported, due to lack of gmcrypto, ask yu.zhang@tongdun.net for the package.")
        
        # check seed value
        self._check_seed_msg()

        # change seed value to bytes
        seed_bytes = self.seed.to_bytes(math.ceil(self.seed.bit_length() / 8), 'big')

        # generate ase encryptor
        encryptor = sm4.new(seed_bytes[:math.ceil(self.key_length / 8)], sm4.MODE_ECB)

        return encryptor, encryptor

    def encrypt(self, data: Union[List, np.ndarray],
                alpha: Optional[int] = None) -> Union[List, np.ndarray]:
        """
        This method mainly encrypt data
        Arg:
            data: list/numpy, encrypt value
            alpha: add multiples to random numbers, just for onetime pad
        Return:
            data / encrypt data
        """
        # encrypt method is not None, encrypt data
        if not self.empty:
            if self.method == CRYPTO_ONETIME_PAD:
                # onetime pad need alpha param
                data = self.en.encrypt(data, alpha=alpha)
            else:
                data = self.en.encrypt(data)

        return data

    def decrypt(self, data: Union[List, np.ndarray],
                alpha: Optional[int] = None) -> Union[List, np.ndarray]:
        """
        This method mainly decrypt data
        Arg:
            data: List/numpy, data that has been encrypted
            alpha: minus multiples to random numbers, just for onetime pad
        Return: decrypt value
        """
        if not self.empty:
            # decrypt data
            if self.method == CRYPTO_ONETIME_PAD:
                # onetime pad need alpha param
                data = self.de.decrypt(data, alpha=alpha)
            else:
                data = self.de.decrypt(data)

        return data

    def carry_out_key(self):
        """
        Get private/public key of encrypt method
        ----

        Return:
             public/private key
        """
        return self.en, self.de

    def load_key(self, key_msg: Dict) -> None:
        """
        This method mainly load key msg
        ----

        Args:
            key_msg: dict, save en/de key messages
        ----

        Returns:
             public/private key
        ----

        **Example**:
        >>> method = "paillier"
        >>> param = {"key_length": 1024}
        >>> ecc = EncryptModel(method, param)
        >>> key_msg = dict()
        >>> key_msg['en'] = ecc.en
        >>> key_msg['de'] = ecc.de
        >>> self.load_key(key_msg)
        """
        if key_msg.get('en') is None or key_msg.get('de') is None:
            raise ValueError('must given encrypt/decrypt object')
        self.en = key_msg.get('en')
        self.de = key_msg.get('de')

    def _check_seed_msg(self) -> None:
        """
        This method check if seed is not exist
        """
        if self.seed is None:
            raise ValueError('diffie hellman seed is not exist')
