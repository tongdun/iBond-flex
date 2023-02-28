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
from typing import List, Tuple, Union, Callable, Dict, Optional

from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.utils import ClassMethodAutoLog
from flex.cores.base_model import BaseModel


class OTINVBaseModel(BaseModel):
    """
    Apply secure intersection protocol of invisible inquiry.
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        Invisible inquiry protocol param inits
        inits of federation information for communication and secure params for security calculation
        algo param for select the filter size.
        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
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

        >>> sec_param = [['aes', {'key_length': 128}], ['ot', {'n': 10, 'k': 1}]]

        >>> algo_param = {}

        >>> OTINVServer(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        # inits communication
        self.id_channel = self.commu.guest2host_single_channel('id')

        # init ot input param
        self.k = self.sec_param.param[1]['k']
        self.n = self.sec_param.param[1]['n']

    def _get_ot(self, k, n, remote_id):
        """
        invoking ot protocol
        Args:
            n: int, message number that server need to provide.
            k: int, message number that client want to search, only support 1.
            remote_id: str, remote ID
        returns:
            ot class
        """
        self.ot = make_ot_protocol(k, n, remote_id)


class OTINVServer(OTINVBaseModel):
    """
    invisible inquiry protocol, server side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        OTINVBaseModel.__init__(self,
                                federal_info=federal_info,
                                sec_param=sec_param,
                                algo_param=algo_param)

        self.remote_id = self.commu.first_guest_id
        self._get_ot(self.k, self.n, self.remote_id)

    def exchange(self, query_fun: Callable[[List[str]],
                 List[str]], *args, **kwargs) -> None:
        """
        get the Confused information what mesassge client want to search and send the encrypt mesassge to client.
        Args:
            query_fun: Callable, function that server data.

        -----

        **Example:**

        >>>def query_fun(in_list):
        >>>result = [str(int(i) * 100) for i in in_list]
        >>>return result
        >>>OTINVServer.exchange(query_fun)
        """
        # server receive id list what client want to search
        ids = self.id_channel.recv()
        self.logger.info('receive id list what client want to search')

        # use query function to find the result of query
        query_result = query_fun(ids)
        self.logger.info('use query function to find the result of query')

        # use ot protocol to send the result of query to client
        self.ot.server(query_result)
        self.logger.info('finish the ot protocol')


class OTINVClient(OTINVBaseModel):
    """
    invisible inquiry protocol, client side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        OTINVBaseModel.__init__(self,
                                federal_info=federal_info,
                                sec_param=sec_param,
                                algo_param=algo_param)

        self.remote_id = self.commu.first_host_id
        self._get_ot(self.k, self.n, self.remote_id)

    def exchange(self, ids: Union[str, List[str]],
                 obfuscator: Callable[[List[str], int], Tuple[List[str], int]],
                 *args, **kwargs) -> Union[str, List[str]]:
        """
        send id for inquiry to server and receive the result of query.
        Args:
            ids: str or list, list of ids or id for inquiry.
            obfuscator: Callable, function that expand k ids to n ids.
        Returns:
            str or list, result of query.

        -----

        **Example:**

        >>>def obfuscator(in_list, n):
        >>>fake_list = [random.randint(0, 100) for i in range(n - len(in_list))]
        >>>index = random.randint(0, n - 1)
        >>>joint_list = fake_list[:index] + in_list + fake_list[index:]
        >>>return joint_list, index
        >>>result = OTINVClient.exchange('50', obfuscator)
        """
        if isinstance(ids, list):
            expended_ids, index = obfuscator(ids, self.n)
        elif isinstance(ids, str):
            expended_ids, index = obfuscator([ids], self.n)
        else:
            raise TypeError(f"Type of input id {type(id)} need to be string or list.")

        # send id to server for inquiry
        self.id_channel.send(expended_ids)
        self.logger.info('send id to server for inquiry')

        # use ot protocol to get result of query
        result = self.ot.client(index)
        self.logger.info('use ot protocol to get result of query')

        return result







