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

from typing import Union

import numpy as np

from ..fpe.api import generate_fpe_encryptor


class IDFilter(object):
    """
    This class is used for creating a simple bloom filter from IDs,
    and providing secure permutation and bit operation on it.
    """
    def __init__(self, log2_len: int = 31, src_filter: Union[None, np.ndarray] = None):
        """
        Create an empty filter or initialize by an existed filter.
        Args:
            log2_len: int, log 2 of filter length, support values between 7 and 64.
            src_filter: None or numpy.ndarray, if None, create a new IDFilter.
        """
        if not 7 <= log2_len <= 64:
            raise ValueError(f"log2_len {log2_len} should be >= 7 and <= 64")
        
        self.log2_bit_length = log2_len
        self.bit_length = 1 << log2_len
        self.modulus = self.bit_length - 1

        if isinstance(src_filter, type(None)):
            self.filter = np.zeros((self.bit_length // 8,), dtype=np.uint8)
        else:
            if not isinstance(src_filter, np.ndarray):
                raise TypeError(f"Type of source filter is {type(src_filter)}, should be an ndarray.")
            if src_filter.dtype == np.bool:
                if len(src_filter) != self.bit_length:
                    raise ValueError(f"Length of src_filter {len(src_filter)} do not match log2_len {log2_len}.")
                src_filter_uint8 = np.packbits(src_filter)
            elif src_filter.dtype == np.uint8:
                if len(src_filter) * 8 != self.bit_length:
                    raise ValueError(f"Length of src_filter {len(src_filter)} do not match log2_len {log2_len}.")
                src_filter_uint8 = src_filter
            else:
                raise TypeError(f"Dtype of src_filter {src_filter.dtype} is not valid.")

            self.filter = src_filter_uint8

    def update(self, ids: Union[list, int]):
        """

        Args:
            ids:

        Returns:

        """
        if isinstance(ids, list):
            temp = np.zeros((self.bit_length,), dtype=np.bool)
            temp[ids] = 1
            self.filter |= np.packbits(temp)
        else:
            index = ids & self.modulus
            quotient, remainder = divmod(index, 8)
            self.filter[quotient] |= 1 << (7 - remainder)

    def export(self, dst_type: str = 'uint8') -> np.ndarray:
        """

        Args:
            dst_type:

        Returns:

        """
        if dst_type == 'uint8':
            return self.filter
        elif dst_type == 'bool':
            return np.unpackbits(self.filter)
        else:
            raise ValueError(f"Dtype {dst_type} is not supported.")

    def __permute(self, secret_key: bytes, mode: 'str'):
        """

        Args:
            secret_key:
            mode:

        Returns:

        """
        if len(secret_key) not in [16, 24, 32]:
            raise ValueError(f"Key length {len(secret_key)} not valid, should be in [16, 24, 32]")

        fpe_encryptor = generate_fpe_encryptor(secret_key, self.log2_bit_length)
        dst_filter = np.zeros((self.bit_length,), dtype=np.bool)

        filter_length = 1 << (self.log2_bit_length - 3)

        for i in range (1 << max(0, (self.log2_bit_length - 3 - 16))):
            start = i * (1 << 16)
            end = min((i + 1) * (1 << 16), filter_length)
            indexes = np.where(np.unpackbits(self.filter[start: end]))[0] + start * 8
            if indexes.size != 0:
                if mode == 'encrypt':
                    dst_indexes = fpe_encryptor.encrypt(indexes).tolist()
                else:
                    dst_indexes = fpe_encryptor.decrypt(indexes).tolist()
                dst_filter[dst_indexes] = 1

        dst_filter = np.packbits(dst_filter)
        return IDFilter(self.log2_bit_length, dst_filter)

    def permute(self, secret_key: bytes):
        """

        Args:
            secret_key:

        Returns:

        """
        result = self.__permute(secret_key, 'encrypt')
        return result

    def inv_permute(self, secret_key: bytes):
        """

        Args:
            secret_key:

        Returns:

        """
        result = self.__permute(secret_key, 'decrypt')
        return result

    def __eq__(self, other):
        """

        Args:
            other:

        Returns:

        """
        result = ~np.bitwise_xor(self.filter, other.filter)
        return IDFilter(self.log2_bit_length, result)

    def __and__(self, other):
        """

        Args:
            other:

        Returns:

        """
        result = np.bitwise_and(self.filter, other.filter)
        return IDFilter(self.log2_bit_length, result)













