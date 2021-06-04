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
from typing import Tuple

import numpy as np


class FMMixin:
    """
    Common for host, guest to calculate forward term and 2-party's embedding term in FM model.
    """

    @staticmethod
    def fm(theta: np.ndarray,
           v: np.ndarray,
           features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward term and 2-party's embedding term in FM model

        Args:
            theta: weight params of FM model's linear term
            v: weight params of FM model's embedding term
            features: origin dataset

        Returns:
            forward term and 2-party's embedding term in FM model
        ----

        **Example:**
        >>> theta = np.array([0.2, 0.3, 0.5])
        >>> v = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> features = np.array([[2, 4, 1], [3, 6, 1], [5, 7, 2], [6, 4, 9]])
        >>> FMMixin.fm(theta, v, features)
        """
        # cal linear term
        f_linear = np.matmul(features, theta)

        # cal embedding term
        sum_square = np.sum(np.matmul(features, v), axis=1) ** 2
        square_sum = np.sum(np.matmul(features ** 2, v ** 2), axis=1)
        f_embedding = 0.5 * (sum_square - square_sum)
        f = f_linear + f_embedding

        # cal 2-party's embedding term
        embedding = np.matmul(features, v)
        return f, embedding
