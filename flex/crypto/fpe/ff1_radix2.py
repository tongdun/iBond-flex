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
from typing import Union

import numpy as np
from Crypto.Cipher import AES

from flex.constants import *


class FF1Radix2Encryptor(object):
    """
    Implements a special case of FF1 mode specified in NIST Special Publication 800-38G,
    where the radix(or base) is 2.
    """

    # TODO: support sm4-ff1
    def __init__(self, key: bytes, n: int, t: bytes = b'', encrypt_algo: str = CRYPTO_AES):
        """
        Init a new FF1Radix2 encryptor.
        Args:
            key: bytes, secret key for encryption, support key length of 16, 24 or 32 bytes.
            n: int, max bit length of input/output(in radix 2).
            t:  bytes, tweak, length of t should be >= 0 and <= maxTlen.
            encrypt_algo: string, symmetric encryption algorithm deployed by encryptor, 'aes' is supported.
        """
        if encrypt_algo not in [CRYPTO_AES]:
            raise NotImplementedError(f"Encryption algorithm {encrypt_algo} is not supported.")

        self.encrypt_algo = encrypt_algo
        self.radix = 2
        self.key = key
        self.n = n
        self.T = t
        self.T_len = len(t)
        self.minlen = math.ceil(math.log(100, self.radix))
        self.maxlen = 64
        self.iters = 10
        self.max_allow_number = self.radix ** self.n - 1

        if not self.minlen <= n <= self.maxlen:
            raise ValueError(f"Parameter n {n} is out of range, accept value >= {self.minlen} and <= {self.maxlen}. ")

        self.ciph = AES.new(self.key, AES.MODE_ECB)

        # step 1
        self.u = math.floor(self.n // 2)
        self.v = self.n - self.u
        self.mask_for_B = int(self.v * '1', 2)

        # step 3
        self.b = math.ceil(math.ceil(self.v * math.log(self.radix, 2)) / 8)

        # step 4
        self.d = 4 * math.ceil(self.b / 4) + 4

        # step 5 + step 6-i(1)
        self.len_P = 16
        # self.len_Q = self.T_len + ((-self.T_len - self.b - 1) % 16) + 1 + self.b
        self.len_Q = 16 * math.ceil((self.T_len + self.b + 1) / 16)
        self.PQ = np.zeros((self.len_P + self.len_Q), dtype=np.uint8)
        self.PQ[0] = 1
        self.PQ[1] = 2
        self.PQ[2] = 1
        self.PQ[3] = (self.radix >> 16) & 0xff
        self.PQ[4] = (self.radix >> 8) & 0xff
        self.PQ[5] = self.radix & 0xff
        self.PQ[6] = 10
        self.PQ[7] = self.u & 0xff

        self.PQ[8] = (n >> 24) & 0xff
        self.PQ[9] = (n >> 16) & 0xff
        self.PQ[10] = (n >> 8) & 0xff
        self.PQ[11] = n & 0xff

        self.PQ[12] = (self.T_len >> 24) & 0xff
        self.PQ[13] = (self.T_len >> 16) & 0xff
        self.PQ[14] = (self.T_len >> 8) & 0xff
        self.PQ[15] = self.T_len & 0xff

        self.PQ[self.len_P: self.len_P + self.T_len] = np.frombuffer(self.T, dtype=np.uint8)
        self.PQ_var_index = self.len_P + self.T_len + (-self.T_len - self.b - 1) % 16

        self.PQ_m = None

        self.radix_vector = np.zeros((16, 2), dtype=np.uint32)
        # i is even --> column 0
        # i is odd --> column 1
        for i in range(math.ceil(self.u / 8)):
            self.radix_vector[-i - 1, 0] = (1 << (8 * i))
        for i in range(math.ceil(self.v / 8)):
            self.radix_vector[-i - 1, 1] = (1 << (8 * i))

        self.m = [self.u, self.v]
        self.mask_for_c = [int(self.u * '1', 2), int(self.v * '1', 2)]
        self.power_for_res = (1 << self.v)

    def _encrypt(self, X: np.ndarray) -> np.ndarray:
        """
        Encryption on np.uint32 or np.uint64
        Args:
            X: np.uint32 or np.uint64. Supported shape is (s, 1)

        Returns: np.uint32 or np.uint64, same type and shape as X.
        """
        s = X.shape[0]

        # step 2
        A = np.right_shift(X, self.v)
        B = np.bitwise_and(X, self.mask_for_B)

        self.PQ_m = np.tile(self.PQ, (s, 1))

        # step 6
        for i in range(10):
            # step 6-i-2
            self.PQ_m[:, self.PQ_var_index] = i
            temp1 = np.frombuffer(B.byteswap().tobytes(), dtype=np.uint8)
            self.PQ_m[:, self.PQ_var_index + 1:] = temp1.reshape(s, -1)[:, -self.b:]

            # step 6-ii
            R = np.zeros((s, 16), dtype=np.uint8)
            for j in range(self.PQ_m.shape[1] // 16):
                R = self.ciph.encrypt((R ^ self.PQ_m[:, 16 * j:16 * (j + 1)]).tobytes())
                R = np.frombuffer(R, dtype=np.uint8).reshape(s, -1)

            # step 6-iii
            # if n <= 128, len(S) <= R
            S = R[:, :self.d]

            index = i % 2

            # step 6-iv
            y = np.matmul(S, self.radix_vector[-self.d:, index])[:, np.newaxis]

            # step 6-v
            # m = self.m[i % 2]

            # step 6-vi, 6-vii
            C = np.bitwise_and(A + y, self.mask_for_c[index])

            # step 6-viii
            A = B

            # step 6-ix
            B = C

        return A * self.power_for_res + B

    def _decrypt(self, X):
        """
        Decryption on np.uint32 or np.uint64
        Args:
            X: np.uint32 or np.uint64. Supported shape is (s, 1)

        Returns: np.uint32 or np.uint64, same type and shape as X.
        """
        s = X.shape[0]

        # step 2
        A = np.right_shift(X, self.v)
        B = np.bitwise_and(X, self.mask_for_B)

        self.PQ_m = np.tile(self.PQ, (s, 1))

        # step 6
        for i in range(9, -1, -1):
            # step 6-i-2
            self.PQ_m[:, self.PQ_var_index] = i
            temp2 = np.frombuffer(A.byteswap().tobytes(), dtype=np.uint8)
            self.PQ_m[:, self.PQ_var_index + 1:] = temp2.reshape(s, -1)[:, -self.b:]

            # step 6-ii
            R = np.zeros((s, 16), dtype=np.uint8)
            for j in range(self.PQ_m.shape[1] // 16):
                R = self.ciph.encrypt((R ^ self.PQ_m[:, 16 * j:16 * (j + 1)]).tobytes())
                R = np.frombuffer(R, dtype=np.uint8).reshape(s, -1)

            # step 6-iii
            # if n <= 128, len(S) <= R
            S = R[:, :self.d]
            index = i % 2

            # step 6-iv
            y = np.matmul(S, self.radix_vector[-self.d:, index])[:, np.newaxis]

            # step 6-v
            # m = self.m[i % 2]

            # step 6-vi, 6-vii
            C = np.bitwise_and(B - y, self.mask_for_c[index])

            # step 6-viii
            B = A

            # step 6-ix
            A = C

        return A * self.power_for_res + B

    def encrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """

        Args:
            x:

        Returns:

        """
        if isinstance(x, int):
            if 0 <= x < 2 ** 32:
                x_ndarray = np.array([x], dtype=np.uint32)
            elif x < 2 ** 64:
                x_ndarray = np.array([x], dtype=np.uint64)
            else:
                raise ValueError(f"Value of input x: {x} not in [0, 2^64).")
        elif isinstance(x, np.ndarray):
            x_ndarray = x
        else:
            raise TypeError(f"Type of input x: {type(x)} not valid.")

        if not np.all((x_ndarray >= 0) & (x_ndarray <= self.max_allow_number)):
            raise ValueError(f"Value of input x: {x} not allowed.")

        x_reshaped = x_ndarray.reshape((x_ndarray.size, 1))
        x_en = self._encrypt(x_reshaped)
        x_en = x_en.reshape(x_ndarray.size)

        if isinstance(x, int):
            return int(x_en[0])
        return x_en

    def decrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        if isinstance(x, int):
            if 0 <= x < 2 ** 32:
                x_ndarray = np.array([x], dtype=np.uint32)
            elif x < 2 ** 64:
                x_ndarray = np.array([x], dtype=np.uint64)
            else:
                raise ValueError(f"Value of input x: {x} not in [0, 2^64).")
        elif isinstance(x, np.ndarray):
            x_ndarray = x
        else:
            raise TypeError(f"Type of input x: {type(x)} not valid.")

        if not np.all((x_ndarray >= 0) & (x_ndarray <= self.max_allow_number)):
            raise ValueError(f"Value of input x: {x} not allowed.")

        x_reshaped = x_ndarray.reshape((x_ndarray.size, 1))
        x_de = self._decrypt(x_reshaped)
        x_de = x_de.reshape(x_ndarray.size)

        if isinstance(x, int):
            return int(x_de[0])
        return x_de
