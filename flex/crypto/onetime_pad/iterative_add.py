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


def iterative_add(a: Union[list, np.ndarray, torch.Tensor], b: Union[list, np.ndarray, torch.Tensor]) -> Union[
    list, np.ndarray, torch.Tensor]:
    """
    Add numpy.ndarray to numpy.ndarray or torch.tensor, both with type int64, or add two nested list.
    """
    if isinstance(a, list):
        return [iterative_add(x, y) for (x, y) in zip(a, b)]
    elif isinstance(a, (np.ndarray, torch.Tensor)):
        return a + b
    else:
        raise TypeError(f"Input type={type(a)} is invalid.")
