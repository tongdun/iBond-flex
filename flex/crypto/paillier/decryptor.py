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
from typing import Union

import numpy as np

from .. import gmpy_math
from .encrypted_number import PaillierEncryptedNumber
from .fixedpoint_number import FixedPointNumber
from .keypair import PaillierPrivateKey, PaillierPublicKey


class PaillierDecryptor(object):
    def __init__(self, pub_key: PaillierPublicKey, priv_key: PaillierPrivateKey):
        self.pub_key = pub_key
        self.priv_key = priv_key

    def __raw_decrypt(self, ciphertext: int) -> int:
        """
        Decrypt a single encrypted number.
        Args:
            ciphertext: int, an encrypted number
        Returns:  
            int. the decrypted plaintext of ciphertext.
        """

        def l_func(x: int, p: int) -> int:
            return (x - 1) // p

        def crt(mp, mq):
            """the Chinese Remainder Theorem as needed for decryption.
            return the solution modulo n=pq.
            """
            return gmpy_math.crt(mp, mq, self.priv_key.p, self.priv_key.q, self.priv_key.q_inverse, self.pub_key.n)

        if not isinstance(ciphertext, int):
            raise TypeError("ciphertext should be an int, not: %s" %
                            type(ciphertext))

        mp = l_func(gmpy_math.powmod(ciphertext,
                                     self.priv_key.p - 1, self.priv_key.psquare),
                    self.priv_key.p) * self.priv_key.hp % self.priv_key.p

        mq = l_func(gmpy_math.powmod(ciphertext,
                                     self.priv_key.q - 1, self.priv_key.qsquare),
                    self.priv_key.q) * self.priv_key.hq % self.priv_key.q

        return crt(mp, mq)

    def _decrypt(self, encrypted_number: PaillierEncryptedNumber) -> Union[int, float]:
        """
        Decrypt a single encrypted number.
        Args:
            encrypted_number: PaillierEncryptedNumber
        Returns:  
            int, float. the decrypted & decoded plaintext of encrypted_number.
        """
        if not isinstance(encrypted_number, PaillierEncryptedNumber):
            raise TypeError("encrypted_number should be an PaillierEncryptedNumber, \
                             not: %s" % type(encrypted_number))

        if self.pub_key != encrypted_number.public_key:
            raise ValueError(
                "encrypted_number was encrypted against a different key!")

        encoded = self.__raw_decrypt(
            encrypted_number.ciphertext(be_secure=False))
        encoded = FixedPointNumber(encoded,
                                   encrypted_number.exponent,
                                   self.pub_key.n,
                                   self.pub_key.max_int)
        decrypt_value = encoded.decode()

        return decrypt_value

    def _decrypt_numpy(self, encrypted_number_numpy: np.ndarray) -> np.ndarray:
        '''
        Decrypt numpy arrays using multi-process
        Args:
            encrypted_number_numpy: np.ndarray[PaillierEncryptedNumber]

        Returns: 
            numpy.array. 
        '''
        s = encrypted_number_numpy.shape
        encrypted_number_flatten = encrypted_number_numpy.reshape(-1)
        # Use multi process to accelerate computaion.
        if len(encrypted_number_flatten) < 64:
            decrypted_value_flatten = np.array(
                [self._decrypt(v) for v in encrypted_number_flatten])
        else:
            process_num = min(multiprocessing.cpu_count(), len(encrypted_number_flatten))
            pool = multiprocessing.Pool(processes=process_num)
            decrypted_value_flatten = np.array(
                pool.map(self._decrypt, encrypted_number_flatten))
            pool.terminate()
        return decrypted_value_flatten.reshape(s)

    def decrypt(self, encrypted_number: Union[np.ndarray, PaillierEncryptedNumber]) -> Union[np.ndarray, int, float]:
        """
        Proxy for encrypt _decrypt and _decrypt_numpy. Use proper method to decrypt input.

        Args:
            encrypted_number: PaillierEncryptedNumber or numpy.array of PaillierEncryptedNumber

        Returns: 
            int, float or numpy.array. 
        """
        if isinstance(encrypted_number, np.ndarray):
            return self._decrypt_numpy(encrypted_number)
        else:
            return self._decrypt(encrypted_number)
