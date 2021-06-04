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

from typing import Union, Dict, List, Optional
import warnings

import numpy as np

from flex.cores.base_model import BaseModel
from flex.utils import ClassMethodAutoLog
from flex.cores.check import CheckMixin
from flex.cores.parser import parse_algo_param, AlgoParamParser
from ..unsuper_bin import EquiFrequentBin


class DTBaseModel(BaseModel):
    """
    Decision Tree binning base model
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        HE_DT_FB protocol param inits
        inits of federation information for communication and secure params for security calculation,
        algo params for algorithm implement.

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo

        ----
        Example:
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
        >>>    "host_party_id": "zhibang-d-014010",
        >>>    "max_bin_num": 6,
        >>>    "frequent_value": 50
        >>> }

        >>> DTBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self,
                           federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()

        # inits data type check
        self.check = CheckMixin

        # algo param inits
        self.algo_param = _self_parse_algo_param(algo_param)

        # custom param inits
        self.init_points = np.array([])  # save init split point msg
        self.split_points = np.array([])  # save bin split msg

        # inits communication
        self.host_id = self.algo_param.host_party_id
        self.label_channel = self.commu.guest2host_id_channel(self.host_id, 'label')
        self.bin_info_channel = self.commu.guest2host_id_channel(self.host_id, 'bin_info')
        self.split_channel = self.commu.guest2host_id_channel(self.host_id, 'split_info')
        self.key_channel = self.commu.guest2host_id_channel(self.host_id, 'key')
        self.judge_channel = self.commu.guest2host_id_channel(self.host_id, 'judge')
        self.judge_tree = self.commu.guest2host_id_channel(self.host_id, 'judge_tree')

    @ClassMethodAutoLog()
    def _check_input(self, feature: np.ndarray) -> None:
        self.check.array_type_check(feature)

    @ClassMethodAutoLog()
    def _gen_node_num(self, num_sample: int) -> None:
        """
        This method generate parameters node_num

        Args:
            num_sample: int, number of samples

        Returns:
             None
        """
        if not hasattr(self.algo_param, 'node_num'):
            self.algo_param.node_num = int(0.05 * num_sample)
            self.logger.info("parameter node_num not in init algo param, then generate it by input feature.")

    @ClassMethodAutoLog()
    def _init_split(self, data: np.ndarray) -> np.ndarray:
        """
        This method to init bin candidate split points

        Args:
            data: bin dataset

        Returns:
             np.array, candidate split points
        ----

        **Example:**
        >>> data = np.array([1, 2, 4, 6, 7, 10])
        >>> self._init_split(data)
        """
        # apply equal frequent to bin split
        bin_obj = EquiFrequentBin(self.algo_param.frequent_value)
        initial_bin = bin_obj.get_split_index(data)

        return np.round(initial_bin, 4)


@ClassMethodAutoLog()
def _self_parse_algo_param(algo_param: Dict) -> AlgoParamParser:
    """
    Parse algorithm parameters
    Arg:
        algo_param: dict, params for algo

    Return:
        object
    ----

    **Example:**
    >>> algo_param = {
    >>>    "host_party_id": "zhibang-d-014010",
    >>>    "max_bin_num": 6,
    >>>    "frequent_value": 50
    >>> }

    >>> _self_parse_algo_param(algo_param)
    """
    # algo param inits
    algo_param = parse_algo_param(algo_param)

    # process param max_bin_nums
    if not hasattr(algo_param, 'host_party_id'):
        raise ValueError(f'must provide host id message')

    # process param max_bin_nums
    if not hasattr(algo_param, 'max_bin_num'):
        algo_param.max_bin_num = 6
        warnings.warn('max_bin_num is not input, has set the default value 6.')

    # frequent_value information
    if not hasattr(algo_param, 'frequent_value'):
        algo_param.frequent_value = 50
        warnings.warn('frequent_value is not input, has set the default value 50.')

    return algo_param


