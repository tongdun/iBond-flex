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

import json
from io import open
from typing import Dict, Optional, List

from flex.utils import ClassMethodAutoLog
from flex.sec_config import SEC_METHOD, SEC_DICT


class FederalInfoParser(object):
    """
    This method mainly set federation info
    """
    @ClassMethodAutoLog()
    def __init__(self, federal_info: Dict):
        """
        This method mainly parser the information of federation parameters
        Args:
            federal_info: dict, federation parameters
        Example:
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

            >>> federal_info = FederalInfoParser(federal_info)
        """
        # get federation information of sever,role, local_id and job_id
        self.server: str = federal_info.get('server')
        session = federal_info.get("session")
        self.role: str = session.get("role")
        self.local_id: str = session.get("local_id")
        self.job_id: str = session.get("job_id")

        # get information of guest, host and coordinator
        federation = federal_info.get("federation")
        self.host = federation.get("host", [])
        self.guest = federation.get("guest", [])
        self.coordinator = federation.get("coordinator", [])

        # guest and host ID combine
        self.guest_host = self.guest + self.host

        # coord and host ID combine
        self.coord_host = self.coordinator + self.host

        # participant messages
        self.other_parties = self.guest + self.host
        if self.local_id in self.other_parties:
            self.other_parties.remove(self.local_id)

        # judge cross_feature/cross_sample
        self.cross_feature = len(self.guest) == 1
        self.cross_sample = len(self.host) == 0


class Dict2Obj(object):
    """
    This method mainly trans dict to object
    """
    @ClassMethodAutoLog()
    def __init__(self, dictionary: Dict):

        self.empty = dictionary is None or len(dictionary) == 0

        if not self.empty:
            # input param is not empty
            for key, value in dictionary.items():

                # set param's key as self variable
                setattr(self, key, value)


class SecParamParser:
    @ClassMethodAutoLog()
    def __init__(self, sec_param: List):
        """
        This method mainly parser secure params
        Arg:
            sec_param: list, save secure params
        return
            None
        Examples:
            >>> sec_param = [['paillier', {'key_length': 1024}],]
        """
        # judge if sec_param is null
        self.empty = sec_param is None or len(sec_param) == 0

        self.method, self.param = [], []
        # sec_param is not null
        if not self.empty:
            for i, sec_value in enumerate(sec_param):
                # determine the encrypt method is supported
                if sec_value[0] not in SEC_METHOD:
                    raise ValueError('Type encrypt method is not support， please check your' +
                                     'encrypt method or SEC_METHOD in file flex/sec_config.py')
                else:
                    # append encrypt method
                    self.method.append(sec_value[0])

                    # append encrypt parameters, add default parameters
                    if len(sec_value) == 1 or len(sec_value[1]) == 0:
                        self.param.append(SEC_DICT[sec_value[0]])
                    else:
                        self.param.append(sec_value[1])

        # check diffie_hellman is in method
        if 'diffie_hellman' in self.method:
            raise ValueError('Type secure method has diffie hellman, please remove')


class AlgoParamParser(Dict2Obj):
    @ClassMethodAutoLog()
    def __init__(self, algo_param):
        super().__init__(algo_param)


@ClassMethodAutoLog()
def parse_federal_info(federal_info: Dict) -> FederalInfoParser:
    """
    This method mainly parse federal's parameters
    Arg:
        federal_info: dict, federal info

    Return:
        object

    Example:
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

        >>> parse_federal_info(federal_info)
    """
    federal_info = FederalInfoParser(federal_info)
    return federal_info


@ClassMethodAutoLog()
def parse_sec_param(sec_param: Optional[List]) -> SecParamParser:
    """
    This method mainly parse secure parameters
    Arg:
        sec_param: dict, params for security calc

    Return
        object

    Example:
        >>> sec_parma = [('paillier', {'key_length':1024}), ('aes', {'key_length':128})]
        >>> sec_param = parse_sec_param(sec_param)
    """
    sec_param = SecParamParser(sec_param)
    return sec_param


@ClassMethodAutoLog()
def parse_algo_param(algo_param: Dict) -> AlgoParamParser:
    """
    This method mainly parse algorithm's parameters
    Arg:
        algo_param: dict, prams for algo

    Return:
        object

    Example:
        >>> algo_param = {
        >>>     'lr': 0.01
        >>> }

        >>> algo_param = parse_algo_param(algo_param)
    """
    algo_param = AlgoParamParser(algo_param)
    return algo_param


@ClassMethodAutoLog()
def load_default_sec_param(path: str) -> SecParamParser:
    """
    This method load sec_param from file
    Arg:
        path: file path
    Return:
        object
    """
    with open(path, 'r') as config:
        sec_param = SecParamParser(json.load(config))

        return sec_param


@ClassMethodAutoLog()
def load_default_algo_param(path: str) -> AlgoParamParser:
    """
    This method load algo_param from file
    Arg:
        path: file path
    Return:
        object
    """
    with open(path, 'r') as config:
        algo_param = AlgoParamParser(json.load(config))

        return algo_param
