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
import pandas as pd
import torch

from flex.utils import ClassMethodAutoLog


class CheckMixin:
    """
    This class mainly give all value/type check method
    """

    @staticmethod
    @ClassMethodAutoLog()
    def list_type_check(data: list):
        if not isinstance(data, list):
            raise TypeError(f'Input data is not list.')

    @staticmethod
    @ClassMethodAutoLog()
    def array_type_check(data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError(f'Input data is not np.ndarray.')

    @staticmethod
    @ClassMethodAutoLog()
    def dict_type_check(data: dict):
        if not isinstance(data, dict):
            raise TypeError(f'Input data is not dict.')

    @staticmethod
    @ClassMethodAutoLog()
    def bool_type_check(data: bool):
        if not isinstance(data, bool):
            raise TypeError(f"Input data's type is not bool.")

    @staticmethod
    @ClassMethodAutoLog()
    def int_type_check(data: int):
        if not isinstance(data, bool):
            raise TypeError(f"Input data's type is not int.")

    @staticmethod
    @ClassMethodAutoLog()
    def float_type_check(data: float):
        if not isinstance(data, float):
            raise TypeError(f"Input data's type is not float.")

    @staticmethod
    @ClassMethodAutoLog()
    def dataframe_type_check(data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f'Input data is not pd.DataFrame.')

    @staticmethod
    @ClassMethodAutoLog()
    def series_type_check(data: pd.Series):
        if not isinstance(data, pd.Series):
            raise TypeError(f'Input data is not pd.Series.')

    @staticmethod
    @ClassMethodAutoLog()
    def tensor_type_check(data: torch.Tensor):
        if not isinstance(data, torch.Tensor):
            raise TypeError(f'Input data is not torch.Tensor.')

    @staticmethod
    @ClassMethodAutoLog()
    def data_dimension_check(data: np.ndarray, dimension: int):
        if len(data.shape) != dimension:
            raise ValueError(f'Input data need to be a {dimension}-D np.ndarray.')

    @staticmethod
    @ClassMethodAutoLog()
    def data_relation_check(left_di, right_di):
        if left_di != right_di:
            raise ValueError(f'Dimensions of input data are not same.')

    @staticmethod
    @ClassMethodAutoLog()
    def multi_type_check(data, type: tuple):
        if not isinstance(data, type):
            raise TypeError(f"Input data's type is not in {type}.")

