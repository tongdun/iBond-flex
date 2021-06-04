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

import math
import hashlib
from typing import Dict, List, Optional
import functools
import gc
from collections import namedtuple

import numpy as np
from Crypto.Cipher import AES
try:
    from gmcrypto import sm4
except ImportError:
    sm4 = None
    print('Warning, gmcrypto is not installed, SM4 will not be available.')

from flex.constants import CRYPTO_AES, CRYPTO_SM4
from flex.cores.base_model import BaseModel
from flex.utils import ClassMethodAutoLog
from flex.cores.check import CheckMixin


class AlignBaseModel(BaseModel):
    """
    secure alignment protocol, base model.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        secure alignment protocol param inits
        inits of federation information for communication and secure params for security calculation

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

        >>> sec_param = [['sm4', {'key_length': 128}]]

        >>> algo_param = { }

        >>> AlignBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        self.remote_id = self.commu.guest_host
        self.local_id = self.commu.local_id

        # inits communication channel
        self.aligned_index_channel = self.commu.coord_broadcast_channel(
            'aligned_index')
        self.aligned_verify_channel = self.commu.coord_broadcast_channel(
            'aligned_verify')

        # data type check
        self.check = CheckMixin

    @ClassMethodAutoLog()
    def _check_input(self, ids: List):
        # input data type check
        self.check.list_type_check(ids)

    @ClassMethodAutoLog()
    def _align(self, ids: List) -> List:
        """
        This method mainly calculation data intersection and send back to all party

        Args:
            ids: list, data set

        Returns:
            data intersection
        -----

        **Example:**
        >>>ids = list(range(1000))
        >>>share = make_protocol(SAL, federal_info, sec_param, algo_param)
        >>>result = share.align(ids)
        """
        # check input data
        self._check_input(ids)
        self.logger.info('complete input data check')

        # do DH key exchange protocol to get the seed
        seed = self._generate_diffie_hellman_seed(
            self.remote_id, self.local_id)
        seed_bytes = seed.to_bytes(math.ceil(seed.bit_length() / 8), 'big')
        self.logger.info('get the seed')

        # choose the crypto method
        if self.pf_ecc.method == CRYPTO_AES:
            assert self.pf_ecc.key_length in [
                128, 196, 256], 'The crypto key length is not support.'
            encryptor = AES.new(seed_bytes[:math.ceil(
                self.pf_ecc.key_length/8)], AES.MODE_ECB)
        elif self.pf_ecc.method == CRYPTO_SM4:
            if sm4 is None:
                raise RuntimeError(
                    "SM4 is not supported, due to lack of gmcrypto, ask yu.zhang@tongdun.net for the package.")
            assert self.pf_ecc.key_length == 128, 'The crypto key length is not support.'
            encryptor = sm4.new(seed_bytes[:math.ceil(
                self.pf_ecc.key_length/8)], sm4.MODE_ECB)
        else:
            ValueError('Input symmetric encryption is not support.')

        # encrypt data set and send it to coordinator to do intersection
        encrypted_ids = list(map(lambda x: encryptor.encrypt(
            hashlib.md5(str(x).encode('utf-8')).digest()), ids))
        self.logger.info('Sending encrypted data to coord')
        self.aligned_index_channel.gather(encrypted_ids)

        # get the intersection index from coord
        self.logger.info('Getting the intersection index from coord')
        intersection_index = self.aligned_index_channel.scatter()
        intersection = [ids[i] for i in intersection_index]

        return intersection

    @ClassMethodAutoLog()
    def _verify(self, ids: List) -> bool:
        """
        This method mainly verify result is data intersection or not
        Args:
            ids: list, data set
        Return:
            True or False

        -----

        **Example:**

        >>>ids = list(range(1000))
        >>>share = make_protocol(SAL, federal_info, sec_param, algo_param)
        >>>result = share.verify(ids)
        """
        # check if samples are already aligned
        joined_ids = ''.join([str(x) for x in ids])
        hash_ids = hashlib.sha256(joined_ids.encode('utf-8')).digest()
        self.aligned_verify_channel.gather(hash_ids)
        self.logger.info(
            'send hash_ids to coord to check data are already aligned')
        return all(self.aligned_verify_channel.broadcast())


class SALCoord(AlignBaseModel):
    """
    secure alignment protocol, coord side.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        AlignBaseModel.__init__(self,
                                federal_info=federal_info,
                                sec_param=sec_param,
                                algo_param=algo_param)

    def align(self, *args, **kwargs) -> None:
        """
        Align the data from host and guest.
        """
        DataPack = namedtuple(
            'DataPack', ['idx', 'data', 'length', 'inter_id'])

        def cal_intersection(x: DataPack, y: DataPack):
            gc.collect()  # Clean memory.
            if isinstance(x, list):
                small_set = x[0]
                _, x_inter_id, y_inter_id = np.intersect1d([small_set.data[i] for i in small_set.inter_id],
                                                           y.data,
                                                           assume_unique=True,
                                                           return_indices=True)
                for item in x:
                    item = item._replace(
                        inter_id=[item.inter_id[i] for i in x_inter_id])

                y = y._replace(inter_id=y_inter_id)
                return [*x, y]
            else:
                _, x_inter_id, y_inter_id = np.intersect1d(x.data,
                                                           y.data,
                                                           assume_unique=True,
                                                           return_indices=True)
                x = x._replace(inter_id=x_inter_id)
                y = y._replace(inter_id=y_inter_id)
                return [x, y]

        # get the data set list from all party
        ids_list = self.aligned_index_channel.gather()
        self.logger.info('get the data set list from all party')
        ids_list = [DataPack(idx, i, len(i), None)
                    for idx, i in enumerate(ids_list)]
        ids_list = sorted(ids_list, key=lambda x: x.length)
        intersection = functools.reduce(cal_intersection, ids_list)
        intersection = sorted(intersection, key=lambda x: x.idx)
        intersection_index = [item.inter_id for item in intersection]
        self.logger.info('calculation data intersection complete')
        # send the intersection index to all party
        self.aligned_index_channel.scatter(intersection_index)
        self.logger.info('send the intersection index to all party')

    def verify(self, *args, **kwargs) -> None:
        """
        Check if samples are already aligned.
        """
        # get the data set list and calc intersection to all party
        hash_ids = self.aligned_verify_channel.gather()
        verify_ids = [hash_ids[0] == x for x in hash_ids[1:]]
        self.aligned_verify_channel.broadcast(verify_ids)


class SALParty(AlignBaseModel):
    """
    secure alignment protocol, party side.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        AlignBaseModel.__init__(self,
                                federal_info=federal_info,
                                sec_param=sec_param,
                                algo_param=algo_param)

        # inits DH input
        self._init_encrypt(share_party=self.remote_id,
                           local_party=self.local_id)

    def align(self, ids: List, *args, **kwargs) -> List:
        """
        This method mainly calculation data intersection and send back to all party

        Args:
            ids: list, data set

        Returns:
            data intersection
        -----

        **Example:**
        >>>ids = list(range(1000))
        >>>share = make_protocol(SAL, federal_info, sec_param, algo_param)
        >>>result = share.align(ids)
        """
        return self._align(ids)

    def verify(self, ids: List, *args, **kwargs) -> bool:
        """
        This method mainly verify result is data intersection or not

        Args:
            ids: list, data set

        Returns:
            True or False
        -----

        **Example:**
        >>>ids = list(range(1000))
        >>>share = make_protocol(SAL, federal_info, sec_param, algo_param)
        >>>result = share.verify(ids)
        """
        return self._verify(ids)
