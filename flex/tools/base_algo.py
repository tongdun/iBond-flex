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

from flex.tools.ionic import commu


class FederalInfoParser(object):
    def __init__(self, federal_info: dict):
        self.server = federal_info.get('server')

        session = federal_info.get("session")
        self.role = session.get("role")
        self.local_id = session.get("local_id")
        self.job_id = session.get("job_id")

        federation = federal_info.get("federation")
        self.host = federation.get("host")
        if not self.host:
            self.host = []
        self.guest = federation.get("guest")
        if not self.guest:
            self.guest = []
        coordinator = federation.get("coordinator")
        if not coordinator:
            self.coordinator = []
        else:
            self.coordinator = coordinator[0] \
                if len(coordinator) == 1 else coordinator

        self.guest_host = self.guest + self.host
        self.other_parties = self.guest + self.host
        if self.local_id in self.other_parties:
            self.other_parties.remove(self.local_id)


class Dict2Obj(object):
    def __init__(self, dictionary):
        if dictionary:
            for key in dictionary:
                setattr(self, key, dictionary[key])


class SecParamParser(Dict2Obj):
    def __init__(self, sec_param):
        super().__init__(sec_param)


class AlgoParamParser(Dict2Obj):
    def __init__(self, algo_param):
        super().__init__(algo_param)


class BaseProtocol(object):
    __slots__ = ['federal_info', 'sec_param', 'algo_param']

    def __init__(self,
                 federal_info: dict,
                 sec_param: dict,
                 algo_param: dict = None):
        self._read_federal_info(federal_info)
        self._read_sec_param(sec_param)
        self._read_algo_param(algo_param)

        commu.init(federal_info)

    def _read_federal_info(self, federal_info):
        self.federal_info = FederalInfoParser(federal_info)

    def _read_sec_param(self, sec_param):
        self.sec_param = SecParamParser(sec_param)

    def _read_algo_param(self, algo_param):
        self.algo_param = AlgoParamParser(algo_param)

    def load_default_sec_param(self, path):
        with open(path, 'r') as config:
            self.sec_param = SecParamParser(json.load(config))

    def load_default_algo_param(self, path):
        with open(path, 'r') as config:
            self.algo_param = AlgoParamParser(json.load(config))
