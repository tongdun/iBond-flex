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

from typing import Dict, Callable, Union, Optional, List

from flex.cores.base_model import BaseModel, send_pubkey, get_pubkey
from flex.cores.check import CheckMixin


class HEMLCoord(BaseModel):
    """
    HE_ML Protocol implementation for Coordinator
    """

    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        HE_ML Protocol coordinator init

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

        >>> sec_param = [['paillier', {'key_length': 1024}], ]

        >>> algo_param = { }

        >>> HEMLCoord(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()
        self.var_chan_g2c = self.commu.guest2coord_single_channel('he_ml_variable')
        self.broadcast_chan = self.commu.coord2host_broadcast_channel('he_ml_broadcast')

        # data type check
        self.check = CheckMixin

    def exchange(self, *args, **kwargs) -> None:
        """
        Main part of protocol
        """
        # Step 1 Receive [r] and pubkey, u_id
        self.pf_ecc.en = self.var_chan_g2c.recv(tag='key')
        r_enc, u_id = self.var_chan_g2c.recv(tag='enc_r-pk-u_id')

        # Step 2 Send pubkey and u_id to hosts
        self.broadcast_chan.broadcast(self.pf_ecc.en, tag='key')
        self.broadcast_chan.broadcast(u_id, tag='pk-u_id')

        # Step 3
        enc_loans_list = self.broadcast_chan.gather(tag='loan_enc')

        # Step 4
        enc_total_loans = sum(enc_loans_list) - r_enc
        self.var_chan_g2c.send(enc_total_loans, tag='enc_total_loans')


class HEMLHost(BaseModel):
    """
    HE_ML Protocol implementation for Coordinator
    """

    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        HE_ML Protocol host init

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example**:
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

        >>> sec_param = [['paillier', {'key_length': 1024}], ]

        >>> algo_param = { }

        >>> HEMLHost(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()
        self.broadcast_chan = self.commu.coord2host_broadcast_channel('he_ml_broadcast')

        # data type check
        self.check = CheckMixin

    def exchange(self, req_loan: Callable, *args, **kwargs) -> None:
        """
        Main part of potocol
        """
        # Step 2 Receive pubkey and u_id
        self.pf_ecc.en = self.broadcast_chan.broadcast('key')
        u_id = self.broadcast_chan.broadcast(tag='pk-u_id')

        # Step 3 Request loan value for u_id, encrypt and send to coordinator
        loan_raw = req_loan(u_id)
        loan_enc = self.pf_ecc.encrypt(loan_raw)
        self.broadcast_chan.gather(loan_enc, tag='loan_enc')


class HEMLGuest(BaseModel):
    """
    HE_ML Protocol implementation for Coordinator
    """

    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        HE_ML Protocol guest init

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example**:
        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "host",
        >>>        "local_id": "zhibang-d-014011",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> sec_param = [['paillier', {'key_length': 1024}], ]

        >>> algo_param = { }

        >>> HEMLGuest(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()
        self.var_chan_g2c = self.commu.guest2coord_single_channel('he_ml_variable')

        # data type check
        self.check = CheckMixin

    def exchange(self, u_id: str, r_raw: Union[int, float],
                 *args, **kwargs) -> int:
        """
        Main part of potocol

        Args:
            user_id: str or int, id of user to be query
            r_raw: threshold value for loan

        Returns:
            1 if excceed r_raw 0 for else
        ----

        **Example:**
        >>> user_id = 'Alice'
        >>> r_raw = 50000
        >>> protocol.exchange(user_id, r_raw)
        """
        # Step 1 Encrypt r, send pubkey and [r] to coordinator
        r_enc = self.pf_ecc.encrypt(r_raw)

        # send_pubkey(self.var_chan_g2c, self.pf_ecc.en)
        self.var_chan_g2c.send(self.pf_ecc.en, tag='key')
        self.var_chan_g2c.send((r_enc, u_id), tag='enc_r-pk-u_id')

        # Step 4
        enc_total_loans = self.var_chan_g2c.recv(tag='enc_total_loans')
        total_loans = self.pf_ecc.decrypt(enc_total_loans)
 
        return int(total_loans >= 0)
