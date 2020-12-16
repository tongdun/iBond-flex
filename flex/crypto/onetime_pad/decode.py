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

MAX_UINT64 = 2 ** 64 - 1
MAX_FLOAT32 = 2 ** (2 ** 7)
MIN_FLOAT32 = -2 ** (2 ** 7)


def int2float(x: np.uint64, reverse_scalar: float) -> np.float32:
    """
    Get the numpy.float32 number back from a numpy.uint64 representation.
    Args:
        x:  numpy.uint64, NumPy rounds to the nearest even value.
        reverse_scalar: float, used to decode int to float.
    Return:
        numpy.float32, [approximate] raw number.
    """
    if x.dtype == np.uint64:
        y = x.astype(np.int64) * reverse_scalar
        if np.any(y > MAX_FLOAT32) or np.any(y < MIN_FLOAT32):
            raise ValueError(
                f"Input x={x} encounter overflow, while type transforming from unsigned int to float32.")
        return y.astype(np.float32)
    else:
        raise TypeError(f"Input x type={x.dtype} is not unit64.")
