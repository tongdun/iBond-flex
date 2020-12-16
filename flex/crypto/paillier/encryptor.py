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

import multiprocessing
from functools import partial

from typing import Dict, Tuple, Union

import numpy as np

from .. import gmpy_math
from .encrypted_number import PaillierEncryptedNumber
from .fixedpoint_number import FixedPointNumber
from .keypair import PaillierPublicKey, PaillierPrivateKey
from .obfuscator import apply_obfuscation
from .raw_encrypt import raw_encrypt


class PaillierEncryptor(object):
    def __init__(self, pub_key: PaillierPublicKey):
        self.pub_key = pub_key

    def __raw_encrypt(self, plaintext: int, random_value: int = None) -> int:
        """
        Encode an int into a encrypted number.
        Args:
            plaintext: int or float, as input number.
            random_value: int, used to do obfuscate if given.

        Returns: 
            int, encrypt number.
        """
        return raw_encrypt(plaintext, self.pub_key, random_value)

    def _encrypt(self, value: Union[int, float], precision: int = None, random_value: int = None) -> PaillierEncryptedNumber:
        """Encode and Paillier encrypt a real number value.

        Args:
            value: int or float, as input number.
            precision: int, the numerical precision
            random_value: int, used to do obfuscate if given.

        Returns: 
            PaillierEncryptedNumber
        """
        encoding = FixedPointNumber.encode(
            value, self.pub_key.n, self.pub_key.max_int, precision)
        obfuscator = random_value or 1
        ciphertext = self.__raw_encrypt(
            encoding.encoding, random_value=obfuscator)
        encryptednumber = PaillierEncryptedNumber(
            self.pub_key, ciphertext, encoding.exponent)
        if random_value is None:
            encryptednumber.apply_obfuscation()

        return encryptednumber

    def _encrypt_numpy(self, values_numpy: np.ndarray, precision: int = None, random_value: int = None) -> np.ndarray:
        """
        Traverse and encrypt every number in input numpy.array.

        Args:
            values_numpy: numpy.array. Should be int or float type.
            precision: int, the numerical precision
            random_value: int, used to do obfuscate if given.

        Returns: 
            Numpy ndarray with the same shape. 
        """
        s = values_numpy.shape
        values_flatten = values_numpy.reshape(-1)
        # Use multi process to accelerate computaion.
        if len(values_flatten) < 64:
            encryptednumber_flatten = np.array(
                [self._encrypt(v, precision, random_value) for v in values_flatten])
        else:
            process_num = min(multiprocessing.cpu_count(), len(values_flatten))
            pool = multiprocessing.Pool(processes=process_num)
            partial_encrypt = partial(
                self._encrypt, precision=precision, random_value=random_value)
            encryptednumber_flatten = np.array(
                pool.map(partial_encrypt, values_flatten))
            pool.terminate()
        return encryptednumber_flatten.reshape(s)

    def encrypt(self, value: Union[np.ndarray, int, float], precision: int = None, random_value: int = None) -> Union[np.ndarray, PaillierEncryptedNumber]:
        """
        Proxy for encrypt _encrypt and _encrypt_numpy. Use proper method to encrypt input.

        Args:
            value: int, float or numpy.array[Should be int or float type].
            precision: int, the numerical precision.
            random_value: int, used to do obfuscate if given.

        Returns: 
            PaillierEncryptedNumber or Numpy ndarray of PaillierEncryptedNumber with the same shape as input. 
        """
        if isinstance(value, np.ndarray):
            return self._encrypt_numpy(value, precision, random_value)
        else:
            return self._encrypt(value, precision, random_value)
