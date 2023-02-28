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

from typing import Union, Dict, Optional, List

import numpy as np
import torch

from flex.cores.base_model import BaseModel
from flex.cores.check import CheckMixin
from flex.utils import ClassMethodAutoLog
from flex.cores.iterative_apply import iterative_divide


class SABaseModel(BaseModel):
    """
        Secure aggregation, base model
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        Secure aggregation protocol param inits
        inits of federation information for communication and secure params for security calculation

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example:**
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

        >>> sec_param = [['onetime_pad', {"key_length": 512}], ]

        >>> algo_param = { }

        >>> SABaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        # inits channel
        self.grad_channel = self.commu.coord_broadcast_channel('grad')

        # data type check
        self.check = CheckMixin


class OTPSAFTCoord(SABaseModel):
    """
        Secure aggregation, coordinate
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        SABaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

    def exchange(self, *args, **kwargs) -> None:
        """
        This method mainly aggregate gradient, then send back to all party

        Returns:
            None, coordinate has none value to return
        ----

        **Examples:**
        >>> OTPSAFTCoord.exchange()
        """
        # Calculate the sum
        theta_sum = sum(self.grad_channel.gather())
        self.grad_channel.broadcast(theta_sum)
        self.logger.info('coordinate send aggregate value to all party')


class OTPSAFTParty(SABaseModel):
    """
        Secure aggregation, party
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        SABaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)
        # inits encrypt
        self._init_encrypt(share_party=self.commu.guest_host,
                           local_party=self.commu.local_id)

    def exchange(self, theta: Union[list, np.ndarray, torch.Tensor],
                 *args, **kwargs) -> Union[list, np.ndarray, torch.Tensor]:
        """
        This method mainly send local gradient to coordinate, and get aggregate results

        Args:
            theta: list、np.ndarray or tensor, a intermediate results，may be a gradient

        Returns:
            aggregate results，and same type as the input.
        ----

        **Example:**
        >>> theta = [np.random.uniform(-1, 1, (2, 4)).astype(np.float32),
        >>>          np.random.uniform(-1, 1, (2, 6)).astype(np.float32)]
        >>> OTPSAFTParty.exchange(theta)
        """
        # encrypt local data, send to coordinate
        enc_theta = self.pf_ecc.encrypt(theta, 1)
        self.grad_channel.gather(enc_theta)
        self.logger.info('party encrypt local data, send to coord')

        theta_sum = self.grad_channel.broadcast()
        avg_theta = iterative_divide(self.pf_ecc.decrypt(theta_sum, 2), len(self.commu.guest_host))
        self.logger.info('party decrypt data, given average result')
        return avg_theta
