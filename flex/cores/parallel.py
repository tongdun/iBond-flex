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

from typing import List, Dict, Callable, Optional, Union
from functools import partial
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from unittest.case import TestCase as tc
import warnings
import psutil

import numpy as np

from flex.utils import FunctionAutoLog
from flex.cores.check import CheckMixin as cm
from flex.sec_config import \
    (FREE_CORE_RATIO,
     FREE_MEMORY_RATIO,
     DATA_LENGTH,
     DEFAULT_CORE,
     DEFAULT_MEMORY_CORE)


@FunctionAutoLog(__file__)
def multi_process(func: Callable, data: List,
                  memory_num: Optional[int] = None,
                  param: Dict = None) -> List:
    """
    This method mainly multi-process encrypt data

    Args:
        func: object, functions
        data: list, encrypt value
        memory_num: int, nums of core can use by memory cost
        param: dict, parameters for functions

    Returns:
        data / encrypt data
    """
    if not isinstance(data, list):
        raise TypeError(f'Input data type is {type(data)}, we just support list')

    # numbers of process limit by core of cpu
    process_num = max(int(multiprocessing.cpu_count() *
                          (1 - FREE_CORE_RATIO)), DEFAULT_CORE)
    # according to memory and ore of cpu determine num og process
    if memory_num is not None:
        process_num = min(memory_num, process_num)

    # Use multi process to accelerate computation.
    if len(data) < DATA_LENGTH * process_num or process_num == 1:
        result = func(data, param)
    else:
        # slide data according to process numbers
        data_flatten = _data_flatten(data, process_num)

        # multiply process
        pool = multiprocessing.Pool(processes=process_num)
        partial_func = partial(func, param=param)
        result = pool.map(partial_func, data_flatten)
        pool.terminate()

        result = sum(result, [])

    return result


@FunctionAutoLog(__file__)
def multi_process_submit(func: Callable, data: List,
                         memory_num: Optional[int] = None,
                         param: Dict = None) -> List:
    """
    This method mainly multi-process encrypt data

    Args:
        func: object, functions
        data: list, encrypt value
        memory_num: int, nums of core can use by memory cost
        param: dict, parameters for functions

    Returns:
        data / encrypt data
    """
    if not isinstance(data, list):
        raise TypeError(f'Input data type is {type(data)}, we just support list')

    # numbers of process limit by core of cpu
    process_num = max(int(multiprocessing.cpu_count() *
                          (1 - FREE_CORE_RATIO)), DEFAULT_CORE)
    # according to memory and ore of cpu determine num og process
    if memory_num is not None:
        process_num = min(memory_num, process_num)
    
    # Use multi process to accelerate computation.
    if len(data) < DATA_LENGTH * process_num \
            or process_num == 1:
        result = func(data, param)
    else:
        # slide data according to process numbers
        data_flatten = _data_flatten(data, process_num)

        # multiply process
        futures = []
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            partial_func = partial(func, param=param)
            for _x in data_flatten:
                futures.append(executor.submit(partial_func, _x))

        result = [r.result() for r in futures]

    return result


def _data_flatten(data: Union[List, np.ndarray],
                  slice_num: int) -> List:
    """
        Slices data according to slice_num
    """
    data_flatten = []
    data_len = len(data) // slice_num if len(data) % slice_num == 0 else len(data) // slice_num + 1
    for i in range(slice_num):
        data_flatten.append(data[i*data_len:(i+1)*data_len])
    data_flatten = [x for x in data_flatten if len(x) > 0]
    return data_flatten


@FunctionAutoLog(__file__)
def multiply(x: np.ndarray,
             y: np.ndarray) -> np.ndarray:
    """
    Apply multiply with a array of encrypt Number
    x/y should be np.ndarray
    if both x and y are array, x dimension must higher than y, shape in the same dimension must be same.

    Args:
        x, np.ndarray of encrypted number
        y, np.ndarray

    Returns:
        x*y, result with the same shape of x.
    ----

    **Example:**
    >>> from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
    >>> pe, pd = generate_paillier_encryptor_decryptor(1024)
    >>> x = np.random.randint(0, 1000, (100,))
    >>> y = np.random.randint(0, 10000, (100,))
    >>> en_x = pe.encrypt(x)
    >>> result = pd.decrypt(multiply(en_x, y))
    >>> np.testing.assert_equal(x*y, result)
    """
    # data type check
    cm.array_type_check(x)
    cm.array_type_check(y)

    # data dimension check
    x_shape = x.shape
    x_shape_len = len(x.shape)
    y_shape_len = len(y.shape)
    min_shape = min(x_shape_len, y_shape_len)

    if abs(x_shape_len - y_shape_len) > 1:
        raise ValueError(f"Only supported the difference between x and y dimensions is less than or equal to 1")
    if x.shape[-min_shape:] != y.shape[-min_shape:]:
        raise ValueError(f"x shape value: {x.shape}, y shape value: {y.shape}, can't multiply")

    process_num = calc_cpu_core()
    if process_num == 1:
        return np.multiply(x, y)

    if x_shape_len > min_shape:
        process_num = calc_cpu_core(x.shape[0])
        if process_num == 1:
            return np.multiply(x, y)
        x = _data_flatten(x, process_num)
        return calculate(x=x, y=y, method=np.multiply, is_sec_flat=False,
                         process_num=process_num, concat=np.vstack)
    else:
        x, y = x.reshape(-1), y.reshape(-1)
        x = _data_flatten(x, process_num)
        y = _data_flatten(y, process_num)
        result = calculate(x=x, y=y, method=np.multiply, is_sec_flat=True,
                           process_num=process_num, concat=np.hstack)
        return result.reshape(x_shape)


@FunctionAutoLog(__file__)
def dot(x: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """
    Apply multiply with a array of encrypt Number
    x/y should be np.ndarray
    if both x and y are array, x dimension must higher than y, shape in the same dimension must be same.

    Args:
        x, np.ndarray
        y, np.ndarray of encrypted number

    Returns:
        x.dot(y).
    ----

    **Example:**
    >>> from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
    >>> pe, pd = generate_paillier_encryptor_decryptor(1024)
    >>> x = np.random.randint(0, 1000, (100,100))
    >>> y = np.random.randint(0, 10000, (100,))
    >>> en_y = pe.encrypt(y)
    >>> result = pd.decrypt(dot(x, en_y))
    >>> np.testing.assert_equal(np.dot(x, y), result)
    """
    # data type check
    cm.array_type_check(x)
    cm.array_type_check(y)

    # data dimension check
    x_shape_len = len(x.shape)
    y_shape_len = len(y.shape)

    # find bigger dimension
    # if x.shape[0] < y.shape[-1]:
    #     return dot(y.T, x.T).T

    if x_shape_len < y_shape_len:
        raise ValueError(f"X dimension must more than Y")
    if x.shape == y.shape and len(set(x.shape)) != 1:
        warnings.warn(f'X shape: {x.shape} equal to y shape: {y.shape}, we transpose x dimension')
    elif x.shape[-1] != y.shape[0] and len(y.shape) > 1:
        raise ValueError(f"x shape value: {x.shape}, y shape value: {y.shape}, can't dot")

    process_num = calc_cpu_core(x.shape[0])
    if process_num == 1 or x_shape_len == 1:
        return np.dot(x, y)

    x = _data_flatten(x, process_num)

    if y_shape_len == 1:
        return calculate(x=x, y=y, method=np.dot, is_sec_flat=False,
                         process_num=process_num, concat=np.hstack)
    else:
        return calculate(x=x, y=y, method=np.dot, is_sec_flat=False,
                         process_num=process_num, concat=np.vstack)


@FunctionAutoLog(__file__)
def add(x: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """
    Apply add with a array of encrypted number
    x and y should be np.ndarray.
    if both x and y are array, they should have same shape.

    Args:
        x, np.ndarray of PaillierEncryptedNumber
        y, np.ndarray.

    Returns:
        x+y, result with the same shape of x.
    ----

    **Example:**
    >>> from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
    >>> pe, pd = generate_paillier_encryptor_decryptor(1024)

    >>> x = np.random.randint(0, 1000, (100,))
    >>> y = np.random.randint(0, 10000, (100,))

    >>> en_x = pe.encrypt(x)
    >>> result = pd.decrypt(add(en_x, y))
    >>> tc.assertAlmostEqual(x+y, result)

    >>> en_y = pe.encrypt(y)
    >>> result = pd.decrypt(add(en_x, en_y))
    >>> np.testing.assert_equal(x+y, result)
    """
    # data type check
    cm.array_type_check(x)
    cm.array_type_check(y)
    x_shape = x.shape

    # data dimension check
    cm.data_relation_check(x.shape, y.shape)

    process_num = calc_cpu_core()
    if process_num == 1:
        return np.add(x, y)

    x, y = x.reshape(-1), y.reshape(-1)
    x = _data_flatten(x, process_num)
    y = _data_flatten(y, process_num)
    result = calculate(x=x, y=y, method=np.add, is_sec_flat=True,
                       process_num=process_num, concat=np.hstack)
    return result.reshape(x_shape)


def calculate(x: Union[np.ndarray, List],
              y: Union[np.ndarray, List],
              method: Callable,
              is_sec_flat: bool,
              process_num: int,
              concat: Callable) -> np.ndarray:
    """
    Apply op defined by f with a array of EncryptedNumber

    Args:
        x: np.array.
        y: np.array.
        method: callable, np.add/np.multiply/np.dot
        is_sec_flat: is/is not the second part is data flatten
        process_num: num of parallel process
        concat: method for data concat
    Returns:
        x[Op]y, result with the same shape of x.
    """
    # multiply process
    futures = []
    with ProcessPoolExecutor(max_workers=process_num) as executor:
        if is_sec_flat:
            for _x, _y in zip(x, y):
                futures.append(executor.submit(method, _x, _y))
        else:
            for _x in x:
                futures.append(executor.submit(method, _x, y))

    for i, r in enumerate(futures):
        if i == 0:
            results = r.result()
        else:
            results = concat((results, r.result()))

    return results


def get_memory_cores(onetime_memory: int) -> int:
    """This method mainly according to onetime process memory to calculating num of cores"""

    # free memory values to limit func memory using
    free_memory = (psutil.virtual_memory().free / 1024 / 1024) * (1 - FREE_MEMORY_RATIO)
    memory_num = max(int(free_memory / onetime_memory), DEFAULT_MEMORY_CORE)
    
    return memory_num


def calc_cpu_core(value: Optional[int] = None):
    # cpu numbers
    process_num = max(int(multiprocessing.cpu_count() *
                          (1 - FREE_CORE_RATIO)), DEFAULT_CORE)
    if value is None:
        return process_num
    else:
        process_num = min(process_num, value)
        return process_num
