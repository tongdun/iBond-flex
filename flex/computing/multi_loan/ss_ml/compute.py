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

from typing import Dict, Optional, List, Union, Callable

import torch

from flex.cores.base_model import BaseModel
from flex.utils import ClassMethodAutoLog
from flex.cores.check import CheckMixin
from flex.computing.ss_computing.compute_coord import SSComputeCoord
from flex.computing.ss_computing.compute_host import SSComputeHost
from flex.computing.ss_computing.compute_guest import SSComputeGuest


class SSMLCoord(BaseModel):
    """
    SS_ML Protocol implementation for Coordinator
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        SS_ML Protocol coordinator init

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example:**
        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "coordinator",
        >>>        "local_id": "zhibang-d-014012",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> sec_param = [['secret_sharing', {'precision': 5}], ]

        >>> algo_param = { }

        >>> SSMLCoord(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self,
                           federal_info=federal_info,
                           sec_param=sec_param)

        # inits channel
        self.chans_sum_shares = self.commu.coord_broadcast_channel('ss_ml_broadcast_HG2C')
        self.chan_req_result = self.commu.guest2coord_single_channel('ss_ml_variable_C2G')

    def exchange(self, *args, **kwargs) -> None:
        """
        SS_ML Protocol coordinator calc
        ----

        **Example:**
        >>> protocol = SSMLCoord(dict(), None, None)
        >>> protocol.exchange()
        """
        # Step 3 Receive shares and compute sum
        lis_enc = self.chans_sum_shares.gather(tag='sum_shares')

        # Step 4 Return to guest if sum of shares greater or equal than 0
        ret = sum(lis_enc) >= 0
        self.chan_req_result.send(ret, tag='ret')


class SSMLHost(BaseModel):
    """
    SS_ML Protocol implementation for Host
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        SS_ML Protocol host init

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example:**
        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "host",
        >>>        "local_id": "zhibang-d-014010",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> sec_param = [['secret_sharing', {'precision': 5}], ]

        >>> algo_param = { }

        >>> SSMLHost(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self,
                           federal_info=federal_info,
                           sec_param=sec_param)

        # inits channel
        self.chans_sum_shares = self.commu.coord_broadcast_channel('ss_ml_broadcast_HG2C')
        self.chans_broadcast_uid = self.commu.guest2host_broadcast_channel('ss_ml_broadcast_G2H')

        self.ss_smpc = SSComputeHost(federal_info, sec_param, algo_param)

    def exchange(self, req_loan: Callable, *args, **kwargs) -> None:
        """
        SS_ML Protocol host calc
        ----

        **Example:**
        >>> protocol = SSMLCoord(dict(), None, None)
        >>> req_loan = lambda user_id: 50
        >>> protocol.exchange(req_loan)
        """
        # Step 1 Get user_id
        user_id = self.chans_broadcast_uid.broadcast(tag='user_id')

        # Step 2 Request value on user_id, and generate shares
        loan_raw = req_loan(user_id)
        shares = self.ss_smpc.share_secrets(torch.as_tensor(loan_raw))

        # Step 3 Receiving sum of shares
        self.chans_sum_shares.gather(sum(shares), tag='sum_shares')


class SSMLGuest(BaseModel):
    """
    SS_ML Protocol implementation for Guest
    """

    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        SS_ML Protocol guest init

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example**:
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

        >>> sec_param = [['secret_sharing', {'precision': 5}], ]

        >>> algo_param = { }

        >>> DOTMULGuest(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self,
                           federal_info=federal_info,
                           sec_param=sec_param)

        # inits channel
        self.chans_sum_shares = self.commu.coord_broadcast_channel('ss_ml_broadcast_HG2C')
        self.chan_req_result = self.commu.guest2coord_single_channel('ss_ml_variable_C2G')
        self.chans_broadcast_uid = self.commu.guest2host_broadcast_channel('ss_ml_broadcast_G2H')

        self.ss_smpc = SSComputeGuest(federal_info, sec_param, algo_param)

        # data type check
        self.check = CheckMixin

    def exchange(self, user_id: Union[str, int],
                 r_raw: Union[int, float], *args, **kwargs):
        """
        SS_ML Protocol host calc

        Args:
            user_id: str or int, id of user to be query
            r_raw: threshold value for loan

        Returns:
            1 if excceed r_raw 0 for else
        ----

        **Example:**
        >>> protocol = SSMLCoord(dict(), None, None)
        >>> user_id = 'Alice'
        >>> r_raw = 50000
        >>> protocol.exchange(user_id, r_raw)
        """
        # Step 0 Checking input
        self.check.multi_type_check(user_id, (str, int))
        self.check.multi_type_check(r_raw, (int, float))
        if isinstance(r_raw, float) and self.ss_smpc.smpc.precision == 0:
            self.logger.info("Warning: float input and precision is 0")

        # Step 1 Prepare input and send user_id to hosts
        self.chans_broadcast_uid.broadcast(var=user_id, tag='user_id')
        tensor = torch.as_tensor(-r_raw)
        shares = self.ss_smpc.share_secrets(tensor)

        # Step 3 Receiving sum of shares
        self.chans_sum_shares.gather(sum(shares), tag='sum_shares')

        # Step 4 Receiving result
        ret = self.chan_req_result.recv(tag='ret')

        return ret.item()
