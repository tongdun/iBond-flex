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

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from flex.cores.check import CheckMixin
from flex.cores.base_model import BaseModel
from flex.utils import ClassMethodAutoLog
from flex.cores.parser import parse_algo_param, AlgoParamParser
from flex.algo_config import IV_FFS_PARAM


class IVBaseModel(BaseModel):
    """
    IV base model
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        IV_FFS protocol param inits
        inits of federation information for communication and secure params for security calculation

        Args:
            federal_info: dict, dict to describe the federation info.
            sec_param: list, list to describe the security parameters.
            algo_param: dict, dict to describe the algorithm parameters.
        ----

        **Example**
        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "guest",
        >>>        "local_id": "zhibang-d-014011",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> sec_param = [['paillier', {"key_length": 1024}], ]

        >>> algo_param = {
        >>>     "iv_thres": 0.02,
        >>>    "adjust_value": 1.0
        >>> }

        >>> IVBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()

        # inits data type check
        self.check = CheckMixin

        # inits algo param
        self.algo_param = self._parse_algo_param(algo_param)

        # inits channel
        self.label_channel = self.commu.guest2host_broadcast_channel('label')
        self.good_bad_nums_channel = self.commu.guest2host_broadcast_channel('good_bad_nums')
        self.iv_woe_value_channel = self.commu.guest2host_broadcast_channel('iv_woe_value')

    @ClassMethodAutoLog()
    def _check_guest_input(self, label: np.ndarray) -> None:
        self.check.array_type_check(label)

    @ClassMethodAutoLog()
    def _check_host_input(self, feature: (np.ndarray, pd.Series), is_category: bool, data_null: bool) -> None:
        self.check.multi_type_check(feature, (np.ndarray, pd.Series))
        self.check.bool_type_check(is_category)
        self.check.bool_type_check(data_null)

    @ClassMethodAutoLog()
    def _parse_algo_param(self, algo_param: Dict) -> AlgoParamParser:
        """
        Parse algorithm parameters

        Arg:
            algo_param: dict, params for algo.

        Return:
            object
        ----

        **Example**
        >>> algo_param = {
        >>>    "iv_thres": 0.02,
        >>>    "adjust_value": 1.0
        >>> }

        >>>parse_algo_param(algo_param)
        """
        # algo param inits
        algo_param = parse_algo_param(algo_param)

        # params
        # iv_thres information
        if not hasattr(algo_param, 'iv_thres'):
            algo_param.iv_thres = IV_FFS_PARAM['iv_thres']
            warnings.warn(f"iv_thres is not input, has set the default value {IV_FFS_PARAM['iv_thres']}.")

        # adjust_value information
        if not hasattr(algo_param, 'adjust_value'):
            algo_param.adjust_value = IV_FFS_PARAM['adjust_value']
            warnings.warn(f"adjust_value is not input, has set the default value {IV_FFS_PARAM['adjust_value']}.")

        return algo_param
