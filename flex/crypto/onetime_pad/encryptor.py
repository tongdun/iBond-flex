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
import torch

from ..csprng.api import generate_csprng_generator
from .ciphertext import OneTimePadCiphertext
from .decode import int2float

MAX_UINT64 = 2 ** 64 - 1
MAX_FLOAT32 = 2 ** (2 ** 7)
MIN_FLOAT32 = -2 ** (2 ** 7)


class OneTimePadEncryptor(object):
    """
    This class can be used to do encryption and decryption for secure mean and related statistics.
    """

    def __init__(self, secret_key: Union[int, bytes], method: str = 'hmac_drbg', exponent: int = 32):
        """
        Init a new instance.
        Args:
            secret_key: int or bytes, used as a seed for generating subsequent padding key.
            method: str, 'hmac_drbg' is the only supporting method by now.
            exponent: int, plaintext will be encoded to plaintext * (2 ** exponent).
        """
        self.__secret_key = secret_key
        self.exponent = exponent
        self.__scalar = 2 ** exponent
        self.__reverse_scalar = 2 ** (-exponent)
        self.__noise = None
        self.__ori_torch = False
        self.__ori_dtype = None

        self.__counter = 0
        self.pad_generator = generate_csprng_generator(secret_key, b'', method)
        self.max_iterations = self.pad_generator.max_iterations

    def __float2int(self, x: np.float32) -> np.uint64:

        """
        Use a numpy.uint64 to represent a numpy.float32 number.
        Args:
            x: numpy.float32, raw number
        Return:
            numpy.uint64, NumPy rounds to the nearest even value.
        """
        y = np.around(x * self.__scalar)
        if np.any(y > MAX_UINT64):
            raise ValueError(
                f"Input x={x} encounter overflow, while type transforming from float32 to unsigned int64.")

        return y.astype(np.uint64)

    def __int2float(self, x: np.uint64) -> np.float32:
        """
        Get the numpy.float32 number back from a numpy.uint64 representation.
        Args:
            x:  numpy.uint64, NumPy rounds to the nearest even value.
        Return:
            numpy.float32, [approximate] raw number
        """
        return int2float(x, self.__reverse_scalar)

    def __get_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Generate numpy.uint64 noise of the same shape with input x using hmac_drbg.
        """

        def bytes2int(bytes_in: bytes) -> int:
            return int.from_bytes(bytes_in, byteorder='big')

        amount = x.size
        random_integers = []
        integers_per_call = 80
        byte_length = 8  # for 64 bit integer

        for _ in range(math.ceil(amount / integers_per_call)):
            tmp = self.pad_generator.generate(num_bytes=byte_length * integers_per_call)
            for i in range(integers_per_call):
                random_integers.append(bytes2int(tmp[byte_length * i:byte_length * (i + 1)]))

        random_integers = random_integers[:amount]
        random_integers = np.array(random_integers).astype(np.uint64)
        return random_integers.reshape(x.shape)

    def __get_hierarchical_noise(self, x: Union[list, np.ndarray, torch.Tensor]) -> \
            Union[list, np.ndarray, torch.Tensor]:
        """
        Generate hierarchical noise of the same structure with input x using hmac_drbg.
        """
        if isinstance(x, list):
            __noise = []
            __ori_dtype = []
            __ori_torch = []
            __ori_shape = []

            for element in x:
                if isinstance(element, list):
                    _noise, _dtype, _torch, _shape = self.__get_hierarchical_noise(element)
                    __noise.append(_noise)
                    __ori_dtype.append(_dtype)
                    __ori_torch.append(_torch)
                    __ori_shape.append(_shape)
                elif isinstance(element, np.ndarray):
                    __noise.append(self.__get_noise(element))
                    __ori_dtype.append(element.dtype)
                    __ori_torch.append(False)
                    __ori_shape.append(element.shape)
                elif isinstance(element, torch.Tensor):
                    element_ndarray = element.detach().numpy()
                    __noise.append(self.__get_noise(element_ndarray))
                    __ori_dtype.append(element_ndarray.dtype)
                    __ori_torch.append(True)
                    __ori_shape.append(element_ndarray.shape)
                else:
                    TypeError(f"Input x type={element.dtype} is not list, np.ndarray or torch.Tensor.")
            return [__noise, __ori_dtype, __ori_torch, __ori_shape]
        elif isinstance(x, np.ndarray):
            __noise = self.__get_noise(x)
            __ori_dtype = x.dtype
            __ori_torch = False
            __ori_shape = x.shape
            return [__noise, __ori_dtype, __ori_torch, __ori_shape]
        elif isinstance(x, torch.Tensor):
            x_ndarray = x.detach().numpy()
            __noise = self.__get_noise(x_ndarray)
            __ori_dtype = x_ndarray.dtype
            __ori_torch = True
            __ori_shape = x_ndarray.shape
            return [__noise, __ori_dtype, __ori_torch, __ori_shape]
        else:
            raise TypeError(f"Input x type={x.dtype} is not list, np.ndarray or torch.Tensor.")

    def add_noise(self, x: np.ndarray, noise: np.ndarray, alpha: int) -> np.ndarray:
        """
        Apply result = x + alpha * noise
        """
        if x.dtype == np.float32:
            x = self.__float2int(x)
        elif x.dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            x = x.astype(np.uint64)
        else:
            raise TypeError(f"x.dtype={x.dtype} is not a valid type.")

        result = x + np.array(alpha, dtype=np.uint64) * noise
        return result

    def remove_noise(self, x: np.ndarray, noise: np.ndarray, obj_dtype: np.dtype, is_obj_torch: bool,
                     alpha: int) -> np.ndarray:
        """
        Apply result = x - alpha * noise
        """
        result = x - np.array(alpha, dtype=np.uint64) * noise
        if obj_dtype == np.float32:
            result = self.__int2float(result)
        else:
            result = result.astype(obj_dtype)
        if is_obj_torch:
            result = torch.from_numpy(result)
        return result

    def _encrypt(self, x: Union[list, np.ndarray, torch.Tensor], noise: Union[list, np.ndarray], alpha: int = 1) -> \
            Union[list, np.ndarray]:
        """
        Add noise to data, using one time pad.
        Args:
            x: numpy.ndarray, torch.Tensor with dtype float32, int32, uint32, int64, uint64,
               or nested list of both types.
            noise: list, numpy.ndarray, with dtype uint64.
            alpha: int, 1 by default.
        Return:
            numpy.uint64 list or numpy.uint64, recursively apply x + alpha*noise.
        """

        if isinstance(x, list):
            result = []

            for i, element in enumerate(x):
                if isinstance(x, list):
                    result.append(self._encrypt(element, noise[i], alpha))
                elif isinstance(element, np.ndarray):
                    padded_element = self.add_noise(element, noise[i], alpha)
                    result.append(padded_element)
                elif isinstance(element, torch.Tensor):
                    element_ndarray = element.detach().numpy()
                    padded_element = self.add_noise(element_ndarray, noise[i], alpha)
                    result.append(padded_element)
                else:
                    raise TypeError(f"x.dtype={element.dtype} is not a valid type.")
            return result
        elif isinstance(x, np.ndarray):
            padded_element = self.add_noise(x, noise, alpha)
            return padded_element
        elif isinstance(x, torch.Tensor):
            element_ndarray = x.detach().numpy()
            padded_element = self.add_noise(element_ndarray, noise, alpha)
            return padded_element
        else:
            raise TypeError(f"x.dtype={x.dtype} is not a valid type.")

    def _decrypt(self, x: Union[list, np.ndarray], noise: Union[list, np.ndarray],
                  obj_dtype: Union[list, np.ndarray, torch.Tensor], is_obj_torch: Union[list, bool], alpha: int = 1) \
            -> Union[list, np.ndarray]:
        """
        Remove noise to recover data, using one time pad.
        Args:
            x: numpy.unit64 or nested list of numpy.uint64.
            noise: numpy.unit64 or nested list of numpy.uint64 the same as x.
            obj_dtype: numpy.float32, numpy.int32, numpy.int64, numpy.uint32, numpy.uint64.
            is_obj_torch: bool or nested list of bool values.
            alpha: int, 1 by default.
        Return:
            numpy.ndarray of original type. recursively apply x - alpha*noise.
        """
        if isinstance(x, list):
            result = []

            for i, element in enumerate(x):
                if isinstance(x, list):
                    result.append(self._decrypt(element, noise[i], obj_dtype[i], is_obj_torch[i], alpha))
                elif isinstance(element, np.ndarray):
                    padded_element = self.remove_noise(element, noise[i], obj_dtype[i], is_obj_torch[i], alpha)
                    result.append(padded_element)
                elif isinstance(element, torch.Tensor):
                    element_ndarray = element.detach().numpy()
                    padded_element = self.remove_noise(element_ndarray, noise[i], obj_dtype[i], is_obj_torch[i],
                                                             alpha)
                    result.append(padded_element)
                else:
                    raise TypeError(f"x.dtype={element.dtype} is not a valid type.")
            return result
        elif isinstance(x, np.ndarray):
            padded_element = self.remove_noise(x, noise, obj_dtype, is_obj_torch, alpha)
            return padded_element
        elif isinstance(x, torch.Tensor):
            element_ndarray = x.detach().numpy()
            padded_element = self.remove_noise(element_ndarray, noise, obj_dtype, is_obj_torch, alpha)
            return padded_element
        else:
            raise TypeError(f"x.dtype={x.dtype} is not a valid type.")

    def encrypt(self, x: Union[list, np.ndarray, torch.Tensor], alpha: int = 1) -> OneTimePadCiphertext:
        """
        Add noise to data, using one time pad.
        Args:
            x: list, numpy.ndarray or torch.Tensor, with dtype float32, int32, int64, uint32 or uint64.
            alpha: int, 1 by default.
        Return:
            OneTimePadCiphertext, result = x + alpha*noise.
        """
        self.__noise, self.__ori_dtype, self.__ori_torch, self.__ori_shape = self.__get_hierarchical_noise(x)
        result = self._encrypt(x, self.__noise, alpha)
        result = OneTimePadCiphertext(result, self.__ori_dtype, self.__ori_torch)
        return result

    def decrypt(self, x: OneTimePadCiphertext, alpha: int = 1) -> Union[list, np.ndarray, torch.Tensor]:
        """
        Remove noise to recover data, using one time pad.
        Args:
            x: numpy.ndarray of np.unit64.
            alpha: int, 1 by default.
        Return:
            numpy.uint64 list or numpy.uint64, result = x - alpha*noise.
        """
        if self.__noise is None:
            raise RuntimeError(
                "Call encrypt first and then call decrypt. Make sure call them in pairs.")
        result = self._decrypt(x.ciphertext, self.__noise, self.__ori_dtype, self.__ori_torch, alpha)
        return result
