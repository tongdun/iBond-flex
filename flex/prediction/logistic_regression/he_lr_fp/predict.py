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
from typing import Dict, List, Optional

import numpy as np

from flex.utils import ClassMethodAutoLog
from flex.cores.check import CheckMixin
from flex.cores.base_model import BaseModel, send_pubkey, get_pubkey


class LRFPBaseModel(BaseModel):
    """
    LR prediction base model to init communication channel and secure params
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        """
        LR prediction protocol param inits
        inits of federation information for communication and secure params for security calculation

        Args:
            federal_info: dict, federal info
            sec_param: dict, params for security calc
            algo_param: dict, params for algo

        -----

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

        >>> sec_param = [['paillier', {"key_length": 1024}], ]

        >>> algo_param = { }

        >>> LRFPBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        # inits channel
        self.key_channel = self.commu.guest_broadcast_channel('fp_predict_pub_key')

        # inits encrypt
        self._init_encrypt()

        # data type check
        self.check = CheckMixin

    @ClassMethodAutoLog()
    def _check_input(self, u: np.ndarray):
        # input data type check
        self.check.array_type_check(u)

        # data dimension check
        self.check.data_dimension_check(u, 1)


class HELRFPCoord(LRFPBaseModel):
    """
    LR gradient update protocol, Guest side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        LRFPBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

        # inits channel
        self.data_channel_h = self.commu.coord2host_broadcast_channel('data_h')
        self.data_channel_g = self.commu.coord2guest_broadcast_channel('data_g')

        # get pub key from guest
        self.pf_ecc.en = get_pubkey(self.key_channel)
        self.logger.info('get pub key from guest')

    def exchange(self, *args, **kwargs) -> None:
        """
        This method mainly get prediction in one item
        -----

        **Example:**
        >>> HELRFPCoord.exchange()
        """
        # receive encrypted u1 from guest
        enc_u1 = self.data_channel_g.gather()
        self.logger.info('receive encrypted u1 from guest')

        # receive encrypted u2 from host and calculate the sum of enc_u2 if more than one host
        enc_u2 = self.data_channel_h.gather()
        self.logger.info('receive encrypted u2 from host')
        if len(enc_u2[0]) == 1:
            sum_enc_u2 = enc_u2[0]
        else:
            sum_enc_u2 = sum(enc_u2)

        # calculate the sum of the intermediate results and send it to guest
        self.data_channel_g.broadcast(enc_u1 + sum_enc_u2)
        self.logger.info('calculate the sum of the intermediate results and send it to guest')


class HELRFPGuest(LRFPBaseModel):
    """
    LR gradient update protocol, Guest side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        LRFPBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

        # inits channel
        self.data_channel_g = self.commu.coord2guest_broadcast_channel('data_g')

        # send pubkey to coord and host
        send_pubkey(self.key_channel, self.pf_ecc.en)
        self.logger.info('guest send pubkey to coord and host')

    @ClassMethodAutoLog()
    def exchange(self, u1: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        """
        This method mainly get prediction in one item

        Args:
            u1: the intermediate results

        Returns:
            prediction of this item
        -----

        **Example:**
        >>> u1 = np.random.uniform(-1, 1, (32,))
        >>> self.exchange(u1)
        """
        # check input data
        self._check_input(u1)
        self.logger.info('guest complete input data check')

        # send encrypted u1 to coord
        enc_u1 = self.pf_ecc.encrypt(u1)
        self.data_channel_g.gather(enc_u1)
        self.logger.info('guest send encrypt data to coord')

        # guest get the sum of the intermediate results and decrypt it
        enc_sum_u = self.data_channel_g.broadcast()
        sum_u = self.pf_ecc.decrypt(enc_sum_u)
        return sum_u


class HELRFPHost(LRFPBaseModel):
    """
    LR gradient update protocol, Host side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        LRFPBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

        # inits channel
        self.data_channel_h = self.commu.coord2host_broadcast_channel('data_h')

        # receive pub key from coord
        self.pf_ecc.en = get_pubkey(self.key_channel)

    @ClassMethodAutoLog()
    def exchange(self, u2: np.ndarray,
                 *args, **kwargs) -> None:
        """
        This method mainly get prediction in one item

        Args:
            u2: the intermediate results
        -----

        **Example:**

        >>> u2 = np.random.uniform(-1, 1, (32,))
        >>> self.exchange(u2)
        """
        # check input data
        self._check_input(u2)
        self.logger.info('host complete input data check')

        # send encrypted u2 to Coord
        enc_u2 = self.pf_ecc.encrypt(u2)
        self.data_channel_h.gather(enc_u2)
        self.logger.info('host send encrypt data to coord')
