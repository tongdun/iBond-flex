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
import torch

from ..onetime_pad.iterative_add import iterative_add
from .decode import int2float

MAX_FLOAT32 = 2 ** (2 ** 7)
MIN_FLOAT32 = -2 ** (2 ** 7)


class OneTimePadCiphertext(object):
    """
    This class provides:
     1. addition between two OneTimePadCiphertext instances,
     2. decode a ciphertext to plaintext.
    """
    def __init__(self, ciphertext: Union[list, np.ndarray],
                 ori_dtype: Union[list, np.float32, np.int32, np.int64, np.uint32, np.uint64],
                 ori_torch: Union[list, bool], exponent: int = 32):

        """
        Init a new OneTimePadCiphertext.
        Args:
            ciphertext: nested list of numpy.ndarray or numpy.ndarray
            ori_dtype: np.float32, np.int32, np.int64, np.uint32, np.uint64, or nested list consisted of these dtypes.
                       ori_dtype represents the plaintext dtypes.
            ori_torch: bool values list or bool value, represents if a element in plaintext is a torch tensor.
            exponent: int, used to encode from or decode to plaintext.
        """
        self.ciphertext = ciphertext
        self.exponent = exponent
        self.__reverse_scaler = 2 ** (-exponent)
        self.__ori_dtype = ori_dtype
        self.__ori_torch = ori_torch

    def __add__(self, other: '__class__') -> '__class__':
        # TODO: make sure ori_dtype, ori_torch are the same
        #       support different exponents
        if isinstance(other, __class__):
            result = iterative_add(self.ciphertext, other.ciphertext)
            result = OneTimePadCiphertext(result, self.__ori_dtype, self.__ori_torch, self.exponent)
        else:
            result = self
        return result

    def __radd__(self, other: '__class__') -> '__class__':
        return self.__add__(other)

    def __int2float(self, x: np.uint64) -> np.float32:
        """
        Get the np.float32 number back from a np.unint64 representation.

        Args:
            x:  np.uint64, NumPy rounds to the nearest even value.

        Return:
            np.float32, [approximate] raw number
        """
        return int2float(x, self.__reverse_scaler)

    def __decode(self, x: Union[list, np.ndarray], obj_dtype: Union[list, np.ndarray], obj_torch: Union[list, bool]):
        if isinstance(x, list):
            result = []
            for i, element in enumerate(x):
                if isinstance(x, list):
                    result.append(self.__decode(element, obj_dtype[i], obj_torch[i]))
                elif isinstance(element, np.ndarray):
                    if obj_dtype == np.float32:
                        decoded_element = self.__int2float(x)
                    else:
                        decoded_element = element.astype(obj_dtype)
                    if obj_torch:
                        decoded_element = torch.from_numpy(decoded_element)
                    result.append(decoded_element)
                else:
                    raise TypeError(f"x.dtype={element.dtype} is not a valid type.")
            return result
        elif isinstance(x, np.ndarray):
            if obj_dtype == np.float32:
                decoded_x = self.__int2float(x)
            else:
                decoded_x = x.astype(obj_dtype)

            if obj_torch:
                decoded_x = torch.from_numpy(decoded_x)
            return decoded_x
        else:
            raise TypeError(f"x.dtype={x.dtype} is not a valid type.")

    def decode(self):
        result = self.__decode(self.ciphertext, self.__ori_dtype, self.__ori_torch)
        return result
