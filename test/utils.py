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
import torch


def almost_equal(x, y, epsilon=1e-4):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (x - y) < epsilon
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape:
        return np.all(x - y < epsilon)
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape:
        return torch.all(x - y < epsilon)
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return [almost_equal(x[i], y[i]) for i in range(len(x))]
    return False

