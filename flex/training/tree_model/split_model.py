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

from copy import deepcopy
from typing import List, Union, Dict, Optional

import numpy as np

from flex.utils import ClassMethodAutoLog


class SplitModel:
    """
    This method mainly fine best split point of all feature
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 gain: str,
                 min_samples_leaf: Optional[int],
                 lambda_: Optional[float]):
        """
        Find the best split point of tree node split

        Args:
            gain: str, method of calculate information gain
            min_samples_leaf: int, threshold of samples ables to split
            lambda_: float, params to control complexity of tree construct

        **Example:**
        >>> min_samples_leaf = 10
        >>> gain = 'grad_hess'
        >>> lambda_ = 0.3
        >>> SplitModel(gain, min_samples_leaf, lambda_)
        """
        self.min_samples_leaf = min_samples_leaf
        self.lambda_ = lambda_

        self.gain = gain
        if self.gain not in ['gini', 'grad_hess']:
            raise ValueError('this gain calculation method is not supported, please enter gini or grad_hess')

        self.gain_funcs = {
            'gini': self._gini_split,
            'grad_hess': self._grad_hess_split
        }

    @ClassMethodAutoLog()
    def calc_best_split(self, data: Dict,
                        is_category: Dict,
                        *args, **kwargs) -> Optional[tuple]:
        """
        Args:
            data: dict, main contain all mess of count/grad/hessian in each feature
            is_category: dict, judge continuous(False)/category(True) of feature

        Reture:
            max gain, best split feature, best split bin point
        **Example:**
        >>> data = {
        >>>     'k1':{
        >>>         'count': np.array([10, 25, 25]),
        >>>         'grad': np.array([0.6, 0.8, -1.2]),
        >>>         'hess': np.array([0.9, 0.8, 0.5])
        >>>     },
        >>>     'k2':{
        >>>         'count': np.array([20, 10, 15, 15]),
        >>>         'grad': np.array([1.1, 1.5, -2.3, -0.5]),
        >>>         'hess': np.array([1.3, -0.8, 2.2, -0.5])
        >>>     }
        >>> }

        >>> is_category = {
        >>>     'k1': False, 'k2': True
        >>> }

        >>> SplitModel.calc_best_split(data, is_category)
        """
        # max gain points params inits
        weight = -np.inf
        max_gain = -np.inf
        best_feature, best_split_point = None, None
        max_left_weight, max_right_weight = -np.inf, -np.inf

        gain_func = self.gain_funcs[self.gain]
        # find best split points
        for i, key in enumerate(data):
            weight, max_bin_gain, bin_index, left_weight, right_weight = gain_func(data[key], is_category[key])
            #print("*******************FLEX DATA IN FOR:", data[key])
            #print("*******************FLEX WEIGHT IN FOR:", weight)

            # judge this term's gain is more than before
            if max_gain < max_bin_gain:
                max_gain = max_bin_gain
                best_feature = key
                best_split_point = bin_index
                max_left_weight, max_right_weight = left_weight, right_weight

        #print("**********************FLEX FINAL WEIGHT:", weight)
        # judge whether satisfied the condition of node split
        if sum(data[list(data.keys())[0]]['count']) < 2 * self.min_samples_leaf:
            return weight, None, None, None, None, None

        # judge all feature's gain is consistent
        if best_feature is None:
            return weight, None, None, None, None, None
        else:
            return weight, max_gain, best_feature, best_split_point, max_left_weight, max_right_weight

    def _grad_hess_split(self,
                         data: Dict,
                         is_category: bool) -> tuple:
        """
        Args:
            data: dict, construct of count/grad/hess,
                count mean sample size in each bin
                grad mean sum of gradients in each bin
                hess mean sum of hessian in each bin
            is_category: bool, judge continuous/category of feature

        Returns:
            best split points of bin
        """
        # if in this feature, only has one bin
        if len(data['count']) < 1:
            return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf
        
        # max bin gain points inits
        max_bin_gain = -np.inf
        best_bin_point = None
        max_left_bin_weight, max_right_bin_weight = -np.inf, -np.inf

        # weight calc
        weight = self._grad_hess_weight(sum(data['grad']), sum(data['hess']))
        #print("*****************************FLEX WEIGHT:", weight)

        # left/right
        count = _data_split(data['count'], is_category)
        grad = _data_split(data['grad'], is_category)
        hess = _data_split(data['hess'], is_category)

        # best split info
        for i in range(len(count[0])):
            # calc grad_hess gain
            bin_gain = 0.5 * (self._grad_hess_gain(grad=grad[0][i], hess=hess[0][i]) +
                              self._grad_hess_gain(grad=grad[1][i], hess=hess[1][i]) -
                              self._grad_hess_gain(grad=sum(data['grad']), hess=sum(data['hess'])))

            # best bin gain condition
            if count[0][i] >= self.min_samples_leaf and count[1][i] >= self.min_samples_leaf \
                    and bin_gain > max_bin_gain:
                max_bin_gain = bin_gain
                best_bin_point = i
                # split weight
                max_left_bin_weight = self._grad_hess_weight(grad[0][i], hess[0][i])
                max_right_bin_weight = self._grad_hess_weight(grad[1][i], hess[1][i])

        return weight, max_bin_gain, best_bin_point, max_left_bin_weight, max_right_bin_weight

    def _gini_split(self,
                    data: Dict,
                    is_category: bool) -> tuple:
        """
        Args:
            data: dict, construct of count/y,
                count mean sample size in each bin
                y mean sum value of label in each bin
            is_category: bool, judge continuous/category of feature

        Returns:
            best split points of bin
        """
        # if in this feature, only has one bin
        if len(data['count']) <= 1:
            return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

        # max bin gain points inits
        max_bin_gain = -np.inf
        best_bin_point = None
        max_left_bin_weight, max_right_bin_weight = -np.inf, -np.inf

        # weight calc
        weight = self._grad_hess_weight(sum(data['count']), sum(data['y']))

        # sum of count,grad,hess of feature
        count = _data_split(data['count'], is_category)
        y = _data_split(data['y'], is_category)

        # find max gain and best split point
        for i in range(len(count[0])):
            # calc gini gain
            bin_gain = self._gini_gain(count=(count[0][i], count[1][i]),
                                       y=(y[0][i], y[1][i]))

            # split result
            left_bin_weight = self._grad_hess_weight(count[0][i], y[0][i])
            right_bin_weight = self._grad_hess_weight(count[1][i], y[1][i])

            # best bin gain condition
            if count[0][i] >= self.min_samples_leaf and count[1][i] >= self.min_samples_leaf \
                    and bin_gain > max_bin_gain:
                max_bin_gain = bin_gain
                best_bin_point = i
                max_left_bin_weight, max_right_bin_weight = left_bin_weight, right_bin_weight

        return weight, max_bin_gain, best_bin_point, max_left_bin_weight, max_right_bin_weight

    def _grad_hess_gain(self, grad: float, hess: float) -> float:
        """
        Calculation of gain by gradient and hessian
        """
        return grad ** 2 / (hess + self.lambda_)

    def _grad_hess_weight(self, grad: float, hess: float) -> float:
        """
        Calculation of weight by gradient and hessian
        """

        #print("*******************************FLEX COMPUTE WEIGHT GRAD:", grad)
        #print("*******************************FLEX COMPUTE WEIGHT HESS:", hess)
        #print("*******************************FLEX COMPUTE WEIGHT LAMBDA:", self.lambda_)
        weight = -grad / (hess + self.lambda_)

        return weight

    @staticmethod
    def _gini_gain(count: tuple, y: tuple) -> float:
        """
        Calculation of gain by gini
        """
        # gain of parent node
        num = count[0] + count[1]
        num_p = y[0] + y[1]
        num_n = num - num_p
        gini_o = 1 - (num_p / num) ** 2 - (num_n / num) ** 2

        # gain of left child
        num_nl = count[0] - y[0]
        gini_l = 1 - (y[0] / count[0]) ** 2 - (num_nl / count[0]) ** 2

        # gain of right child
        num_nr = count[1] - y[1]
        gini_r = 1 - (y[1] / count[1]) ** 2 - (num_nr / count[1]) ** 2

        # left/right split gain
        gini_lr = (count[0] / num) * gini_l + (count[1] / num) * gini_r

        # final gain
        bin_gain = gini_o - gini_lr

        return bin_gain

    @staticmethod
    def _gini_weight(count: int, y: int) -> int:
        """
        Calculation of node weight by gini
        """
        # percentage of value labeled 1 in node
        value = y / count

        # predicted results
        weight = 1 if value >= 0.5 else 0

        return weight


def _data_split(data: np.ndarray,
                is_category: bool) -> tuple:
    """
    This method mainly calc data split to left/right node mess

    Args:
        data: array, data need to split
        is_category: bool, judge continuous/category of feature

    Return:
        left/right data split message
    ----

    **Example**:
    >>> data = np.array([20, 15, 30, 18])
    >>> is_category = False

    >>> left_data, right_data = _data_split(data, is_category)
    """
    # data sum
    sum_data = sum(data)

    # left node mess calc
    if is_category:
        # category feature
        left_data = deepcopy(data)
    else:
        # continuous feature
        left_data = np.cumsum(data).tolist()

    # right node mess calc
    right_data = [sum_data - x for x in left_data]

    return left_data, right_data
