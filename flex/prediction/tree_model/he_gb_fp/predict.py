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

from typing import Optional, Dict, List, Union
import copy
import os

import numpy as np

from flex.cores.base_model import BaseModel
from flex.utils import ClassMethodAutoLog
from flex.cores.commu_model import make_raw_broadcast, make_broadcast_channel

class NodeBaseModel(BaseModel):
    """
        Hetero tree node split
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        Tree node predict protocol
        inits of federation information for communication
        secure params for security calculation
        algorithm parameters for

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

        >>> sec_param = None

        >>> algo_param = None
        """
        BaseModel.__init__(self, federal_info, sec_param)
        self._init_encrypt()



class NodeFPParty(NodeBaseModel):
    """
    Tree model predict protocol, mainly used for ID message exchange
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        NodeBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

        self.party_id = None

    def exchange(self, party_id: str,
                 left_data_id: Optional[Union[np.ndarray, List]] = None,
                 tag: str = '*', *args, **kwargs) -> Optional[Union[np.ndarray, List]]:
        """
        This method mainly exchange left node ID msg

        Args:
            party_id: str, node message saved party
            left_data_id: array, node message
            tag: str, run time name for commu

        Returns:
            ID message/None, owner party send ID msg and return None, user get ID msg and return
        ----

        **Example**
        >>> party_id = 'zhibang-d-014010'
        >>> left_data_id = ['aa', 'bb', 'cc', 'dd']
        >>> NodePFParty.exchange(party_id, left_data_id)
        """
        self.party_id = party_id

        # channel inits
        id_channel = self._build_channel(channel_name='id')

        if self.party_id == self.local_id:
            # owner send ID msg
            id_channel.broadcast(left_data_id, tag=tag)

        else:
            # user get ID msg and return
            left_data_id = id_channel.broadcast(tag=tag)
            return left_data_id

    @ClassMethodAutoLog()
    def _build_channel(self, channel_name: str,
                       *args, **kwargs) -> make_broadcast_channel:
        """
        This method mainly build channel beyond all parties

        Args:
            channel_name, str

        Returns:
            channel object
        ----

        **Example**
        >>> channel = NodePFParty._build_channel()
        """
        channel = make_raw_broadcast(channel_name=channel_name,
                                     root=self.party_id,
                                     remote_group=self.group_id,
                                     job_id=self.job_id)
        return channel

    @property
    def group_id(self) -> List:
        group_id = copy.deepcopy(self.commu.federal_info.guest_host)
        group_id.remove(self.party_id)
        return group_id

    @property
    def local_id(self) -> str:
        return self.commu.federal_info.local_id
