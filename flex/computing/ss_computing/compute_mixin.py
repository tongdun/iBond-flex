"""SS_COMPUTE Protocol
"""
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
from typing import Union, Dict, Optional, List

import numpy as np
import torch

from flex.cores.check import CheckMixin
from flex.crypto.smpc.smpc_protocol.secure_nn import secureNNTH as SMPC
from flex.crypto.smpc.smpc_protocol.fixed_precision_tensor import make_fixed_precision_tensor, \
    decode_tensor
from flex.utils import ClassMethodAutoLog
from flex.constants import DEFAULT_MPC_PRECISION


class Mixin:
    """
    Common functions
    """

    @staticmethod
    def _make_mpc_conf(federal_info: Dict, sec_param: Optional[List]):
        """
        Generate mpc conf from federal_info

        Args:
            federal_info: dict, federal_info used in the main Protocol
            algo_param: dict: parameters for the mpc

        Returns:
            mpc_conf: dict, configuration used to initlize mpc instance
        ----

        **Example:**
        >>> federal_info = {
        >>>    "session": {
        >>>        "role": "guest",
        >>>        "local_id": "td-001",
        >>>        "job_id": "test_abc",
        >>>    }
        >>>    "federation": {
        >>>        "host": ["td-005"],
        >>>        "guest": ["td-001"],
        >>>        "coordinator": ["td-006"],
        >>>    }
        >>> }

        >>> algo_param = {
        >>>    "mpc_precision": 3,
        >>> }
        >>> {"GENERAL": {
        >>>     "NUM_PARTY": 2,
        >>>     "PRECISION": 3,
        >>>     "FIELD": 18446744073709551616,},
        >>>  "ADDRESSBOOK": {
        >>>     0: "td-005, PARTY"
        >>>     1: "td-001, PARTY"
        >>>     2: "td-006, ARBITER"}
        >>> }
        """
        mpc_conf = dict()
        mpc_conf['GENERAL'] = dict()
        mpc_conf['GENERAL']['NUM_PARTY'] = len(federal_info.guest_host)
        mpc_precision = -1
        for algo_param in sec_param:
            if algo_param[0] == 'secret_sharing' and 'precision' in algo_param[1]:
                mpc_precision = algo_param[1]['precision']
        if mpc_precision < 0:
            mpc_precision = DEFAULT_MPC_PRECISION
        mpc_conf['GENERAL']['PRECISION'] = mpc_precision

        wid = 0
        mpc_conf['ADDRESSBOOK'] = dict()
        for item in federal_info.host:
            mpc_conf['ADDRESSBOOK'][str(wid)] = f'{item}, PARTY'
            wid += 1
        for item in federal_info.guest:
            mpc_conf['ADDRESSBOOK'][str(wid)] = f'{item}, PARTY'
            wid += 1
        # only one coordinator in SMPC
        mpc_conf['ADDRESSBOOK'][str(wid)] = f'{federal_info.coordinator[0]}, ARBITER'

        return mpc_conf

    @staticmethod
    def _mapping_to_mpc(mpc_conf: Dict):
        """
        Generate reverse_mapping of name to mpc_id

        Args:
            mpc_conf: dict, configuration used to initlize mpc instance

        Returns:
            reverse_mapping: dict, the mapping
        """
        reverse_mapping = dict()
        for mpc_id in mpc_conf['ADDRESSBOOK']:
            name = mpc_conf['ADDRESSBOOK'][mpc_id].split(',')[0]
            reverse_mapping[name] = mpc_id

        return reverse_mapping

    def _check_input(self, mat_raw: Union[torch.Tensor, np.ndarray]) -> tuple:
        """
        Check fedded input
        if input is np.ndarray, convert to torch.tensor

        Args:
            mat_raw: Union[np.ndarray, torch.tensor], input tensor

        Returns:
            mat_raw: torch.tensor, convert to torch.tensor if input is np.ndarray
            input_type: [torch.tensor, numpy.ndarray], input type
            input_dtype: [torch.float, torch.double, touch.half, torch.unit8, torch.int8, torch.short, torch.int,
                          torch.long, torch.bool, numpy.bool, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                          numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,numpy.float32, numpy.float64],
                          dtype of input
        ----

        **Example:**
        >>> mat_raw = np.int32([1,2,8])
        >>> torch.tensor([1, 2, 8], dtype=torch.int), np.ndarray, np.int32
        """
        input_type = type(mat_raw)
        input_dtype = mat_raw.dtype
        if isinstance(mat_raw, np.ndarray):
            mat_raw = torch.from_numpy(mat_raw)
        if not torch.is_tensor(mat_raw):
            raise TypeError('Input type should either be numpy.ndarray or torch.tensor')
        return mat_raw, input_type, input_dtype
