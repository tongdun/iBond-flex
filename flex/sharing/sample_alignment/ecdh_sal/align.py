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

import hashlib
import secrets
from typing import Dict, List, Tuple, Optional
import time
import gc

import numpy as np
from gmpy2 import f_mod, mul

from flex.crypto.ecc.ed25519 import ed25519_donna as edcurve
from flex.cores.base_model import BaseModel
from flex.cores.check import CheckMixin
from flex.cores.parallel import multi_process, get_memory_cores
from flex.utils import ClassMethodAutoLog
from flex.memory_config import ecdh_memory


class ECDHSalBaseModel(BaseModel):
    """
    The secure align protocol based on Elliptic curve Diffie-Hellman crypto.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        ecdh_sal protocol param inits
        inits of federation information for communication and secure params for security calculation

        Args:
            federal_info: dict, dict to describe the federation info.
            sec_param: list, list to describe the security parameters.
            algo_param: dict, dict to describe the algorithm parameters.
        ----

        **Example**
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

        >>> sec_param = [ ]

        >>> algo_param = { }

        >>> ECDHSalBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()

        # inits data type check
        self.check = CheckMixin

        # get a random int
        self.secrets_generator = secrets.SystemRandom()
        self.prime = edcurve.n
        self.rand = self.secrets_generator.randint(1, self.prime - 1)
        self.G = edcurve.g

        # inits channel
        self.encrypted_id_channel = self.commu.guest2host_single_channel('encrypted_id')
        self.double_encrypted_id_channel = self.commu.guest2host_single_channel('double_encrypted_id')

    @ClassMethodAutoLog()
    def _check_input(self, ids: List):
        # input data type check
        self.check.list_type_check(ids)


class ECDHSalGuest(ECDHSalBaseModel):
    """
        Federated data sharing: get aligned ids.
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        ECDHSalBaseModel.__init__(self,
                                  federal_info=federal_info,
                                  sec_param=sec_param,
                                  algo_param=algo_param)

    @ClassMethodAutoLog()
    def align(self, ids: List, *args, **kwargs) -> List:
        """
        Use ECDH method to align the data from host and guest.

        Args:
            ids: list, data set

        Returns:
            data intersection
        -----

        **Example:**
        >>> ids = list(range(1000))
        >>> share = make_protocol(ECDH_SAL, federal_info, sec_param, algo_param)
        >>> result = share.align(ids)
        """
        # check input data
        self._check_input(ids)
        self.logger.info('complete input data check')

        # get memory using
        onetime_memory = get_memory_param(ids)
        memory_num = get_memory_cores(onetime_memory)

        # params for multi-process
        param = dict()
        param['rand'] = self.rand
        param['prime'] = self.prime
        param['G'] = self.G

        time1 = time.time()
        # cpu_nums = multiprocessing.cpu_count()

        power_guest_list = multi_process(func=_apply_hash_multiply, memory_num=memory_num,
                                         data=ids, param=param)
        self.logger.info('Guest use ECDH method to get the first encrypted IDs.')

        power_host_list = self.encrypted_id_channel.swap(power_guest_list)
        del power_guest_list
        gc.collect()
        self.logger.info('Guest exchange the first encrypted IDs with Host.')

        double_power_host = multi_process(func=_apply_multiply, memory_num=memory_num,
                                          data=power_host_list, param=param)
        del power_host_list
        gc.collect()
        self.logger.info('Guest use ECDH method to get the second encrypted IDs.')

        double_power_guest = self.double_encrypted_id_channel.swap(double_power_host)
        self.logger.info('Guest exchange the second encrypted IDs with Host.')

        hashed_double_power_guest = _parser_point(data=double_power_guest, param=param)
        del double_power_guest
        gc.collect()
        hashed_double_power_host = _parser_point(data=double_power_host, param=param)
        del double_power_host
        gc.collect()
        _, guest_ids, _ = np.intersect1d(hashed_double_power_guest, hashed_double_power_host, return_indices=True)
        self.logger.info('Guest get the aligned IDs.')

        return (np.array(ids)[guest_ids]).tolist()


class ECDHSalHost(ECDHSalBaseModel):
    """
        Secure alignment protocol based ECDH method, host side.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        ECDHSalBaseModel.__init__(self,
                                  federal_info=federal_info,
                                  sec_param=sec_param,
                                  algo_param=algo_param)

    @ClassMethodAutoLog()
    def align(self, ids: List, *args, **kwargs) -> List:
        """
        Use ECDH method to align the data from host and guest.

        Args:
            ids: list, data set

        Returns:
            data intersection
        -----

        **Example:**
        >>> ids = list(range(1000))
        >>> share = make_protocol(ECDH_SAL, federal_info, sec_param, algo_param)
        >>> result = share.align(ids)
        """
        time0 = time.time()
        # check input data
        self._check_input(ids)
        self.logger.info('complete input data check')

        # get memory using
        onetime_memory = get_memory_param(ids)
        memory_num = get_memory_cores(onetime_memory)

        # params for multi-process
        param = dict()
        param['rand'] = self.rand
        param['prime'] = self.prime
        param['G'] = self.G

        power_host_list = multi_process(func=_apply_hash_multiply, memory_num=memory_num,
                                        data=ids, param=param)
        self.logger.info('Host use ECDH method to get the first encrypted IDs.')

        power_guest_list = self.encrypted_id_channel.swap(power_host_list)
        del power_host_list
        gc.collect()
        self.logger.info('Host exchange the first encrypted IDs with Guest.')

        double_power_guest = multi_process(func=_apply_multiply, memory_num=memory_num,
                                           data=power_guest_list, param=param)
        del power_guest_list
        gc.collect()
        self.logger.info('Host use ECDH method to get the second encrypted IDs.')

        double_power_host = self.double_encrypted_id_channel.swap(double_power_guest)
        self.logger.info('Host exchange the second encrypted IDs with Guest.')

        hashed_double_power_guest = _parser_point(data=double_power_guest, param=param)
        del double_power_guest
        gc.collect()
        hashed_double_power_host = _parser_point(data=double_power_host, param=param)
        del double_power_host
        gc.collect()
        _, host_ids, _ = np.intersect1d(hashed_double_power_host, hashed_double_power_guest, return_indices=True)
        self.logger.info('Host get the aligned IDs.')

        return (np.array(ids)[host_ids]).tolist()


def _apply_multiply(data: List[Tuple], param: Dict,
                    *args, **kwargs) -> List[Tuple]:
    
    return list(map(lambda x: edcurve.mul(x, param['rand']), data))


def _apply_hash_multiply(data: List[str], param: Dict,
                         *args, **kwargs) -> List[Tuple]:
    data = list(map(lambda x: int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16), data))
    data = list(map(lambda x: edcurve.mul(param['G'], int(f_mod(mul(x, param['rand']), param['prime']))), data))
    return data


def _parser_point(data: List[Tuple],
                  *args, **kwargs) -> np.ndarray:
    hashed_ids = list(map(lambda x: hashlib.md5((" ".join(map(str, x))).encode('utf-8')).hexdigest(), data))
    return np.asarray(hashed_ids)


def get_memory_param(ids: List) -> int:
    """Memory use for ecdh which level by ids"""
    return ecdh_memory(len(ids))

