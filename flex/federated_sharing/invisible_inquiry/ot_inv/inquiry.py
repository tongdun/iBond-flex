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

from typing import List, Tuple, Union, Callable
from os import path

from flex.tools.base_algo import BaseProtocol
from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.tools.ionic import VariableChannel


class Mixin:
    def _make_channel(self, remote_id):
        self.var_chan = VariableChannel(name='invisible_inquiry_variable',
                                        remote_id=remote_id)

    def _get_ot(self, k, n, remote_id):
        self.ot = make_ot_protocol(k, n, remote_id)


class OTINVServer(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict = None, algo_param: dict = None):
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        super().__init__(federal_info, sec_param, algo_param)
        self.remote_id = self.federal_info.other_parties[0]
        self.k = self.algo_param.k
        self.n = self.algo_param.n
        self._get_ot(self.k, self.n, self.remote_id)
        self._make_channel(self.remote_id)

    def exchange(self, query_fun: Callable[[List[str]], List[str]]) -> None:
        ids = self.var_chan.recv(tag='expended_ids')
        query_result = query_fun(ids)
        self.ot.server(query_result)
        return


class OTINVClient(BaseProtocol, Mixin):
    def __init__(self, federal_info: dict, sec_param: dict = None, algo_param: dict = None):
        if sec_param is not None:
            self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        super().__init__(federal_info, sec_param, algo_param)
        self.remote_id = self.federal_info.other_parties[0]
        self.k = self.algo_param.k
        self.n = self.algo_param.n
        self._get_ot(self.k, self.n, self.remote_id)
        self._make_channel(self.remote_id)

    def exchange(self, ids: Union[str, List[str]],
                 obfuscator: Callable[[List[str], int], Tuple[List[str], int]]) -> Union[str, List[str]]:
        """

        Args:
            ids: str or list, list of ids or id for inquiry.
            obfuscator: Callable, function that expand k ids to n ids.

        Returns: str or list, result of query.

        """
        if isinstance(ids, list):
            expended_ids, index = obfuscator(ids, self.n)
        elif isinstance(ids, str):
            expended_ids, index = obfuscator([ids], self.n)
        else:
            raise TypeError(f"Type of input id {type(id)} need to be string or list.")

        self.var_chan.send(expended_ids, tag='expended_ids')
        result = self.ot.client(index)
        return result







