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
from concurrent.futures import ProcessPoolExecutor


def mul(x: np.ndarray, y: Union[np.ndarray, float, int]) -> np.ndarray:
    """
    Apply multiply with a array of PaillierEncryptedNumber
    One of x and y should be np.ndarray.
    if both x and y are array, they should have same shape.

    Args:
        x, np.ndarray of PaillierEncryptedNumber
        y, np.ndarray, float, int. 
        !!! Attention y only contains scalar type, PaillierEncryptedNumber not allowed. !!!

    Returns:
        x*y, result with the same shape of x.

    Example:
    >>> x = np.random.randint(0, 1000, (100,))
    >>> y = np.random.randint(0, 10000, (100,))
    >>> en_x = pe.encrypt(x)
    >>> result = pd.decrypt(parallel_ops.mul(en_x, y))
    >>> assertAlmostEqual(x*y, result)
    """
    return calculate(x, y, 'mul')


def add(x: np.ndarray, y: Union[np.ndarray, float, int]) -> np.ndarray:
    """
    Apply add with a array of PaillierEncryptedNumber
    One of x and y should be np.ndarray.
    if both x and y are array, they should have same shape.

    Args:
        x, np.ndarray of PaillierEncryptedNumber
        y, np.ndarray, float, int.

    Returns:
        x+y, result with the same shape of x.

    Example:
    >>> x = np.random.randint(0, 1000, (100,))
    >>> y = np.random.randint(0, 10000, (100,))

    >>> en_x = pe.encrypt(x)
    >>> result = pd.decrypt(parallel_ops.add(en_x, y))
    >>> assertAlmostEqual(x*y, result)

    >>> en_y = pe.encrypt(y)
    >>> result = pd.decrypt(parallel_ops.add(en_x, en_y))
    >>> assertAlmostEqual(x*y, result)
    """
    return calculate(x, y, 'add')


def _add(x, y):
    """
    Do not change the position of this function, multi-
    process executer will hang if use lambda or inner function.
    """
    return x+y


def _mul(x, y):
    """
    Do not change the position of this function, multi-
    process executer will hang if use lambda or inner function.
    """
    return x*y


def calculate(x: np.ndarray, y: Union[np.ndarray, float, int], method: str) -> np.ndarray:
    """
    Apply op defined by f with a array of PaillierEncryptedNumber

    Args:
        x, np.ndarray, float, int.
        y, np.ndarray, float, int.
        method, str, 'add' or 'mul'

    Returns:
        x[Op]y, result with the same shape of x.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(msg=f"{type(x)} * {type(y)} not supported")

    if method == 'add':
        func = _add
    elif method == 'mul':
        func = _mul
    else:
        raise NotImplemented

    futures = []
    with ProcessPoolExecutor() as executor:
        with np.nditer(x, flags=["refs_ok"], op_flags=["readonly"]) as x_iter:
            if isinstance(y, (int, float)):
                for _x in x_iter:
                    futures.append(executor.submit(func, _x, y))

            if isinstance(y, np.ndarray):
                if x.shape != y.shape:
                    raise ValueError(msg=f"{x.shape} != {y.shape}")

            with np.nditer(y, flags=["refs_ok"], op_flags=["readonly"]) as y_iter:
                for _x, _y in zip(x_iter, y_iter):
                    futures.append(executor.submit(func, _x, _y))

    result = np.array([r.result() for r in futures])
    return result.reshape(x.shape)
