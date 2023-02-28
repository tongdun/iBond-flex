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


def sigmoid(input: np.ndarray) -> np.ndarray:
    """
    This is a sigmoid function.

    Args:
        input: shape (n,m) numpy array. Input data to calc sigmoid result.

    Returns:
        sigmoid: shape (n,m) numpy array, the returned values are always floats. The sigmoid result.

    -----

    **Examples:**

    >>> from flex.utils.activations import sigmoid
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> sigmoid(input)

    >>> # Output:
    >>> array([0.73105858, 0.88079708, 0.95257413, 0.98201379, 0.99330715, \
        0.99752738, 0.99908895, 0.99966465, 0.99966465, 0.99908895, \
        0.99752738, 0.99330715, 0.98201379, 0.95257413, 0.88079708, \
        0.73105858])
    """
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    out = 0.5*np.ones(x.shape)

    # non negative x
    non_negative_indices = (x >= 0)
    z = np.exp(-x[non_negative_indices])
    out[non_negative_indices] = 1.0 / (1.0 + z)

    # negative x
    negative_indices = (x < 0)
    z = np.exp(x[negative_indices])
    out[negative_indices] = z / (1.0 + z)

    return out