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

import random
import hashlib
from typing import Dict, List, Optional, Union
import gc

import numpy as np
from gmpy2 import f_mod, mul, powmod, divm, mpz

from flex.cores.base_model import BaseModel
from flex.cores.parallel import multi_process
from flex.cores.check import CheckMixin
from flex.utils import ClassMethodAutoLog

import time


class RSASalBaseModel(BaseModel):
    """
    The secure align protocol based on RSA crypto.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        rsa_sal protocol param inits
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

        >>> sec_param = [['aes', {"key_length": 2048}]]

        >>> algo_param = { }

        >>> RSASalBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()

        # inits data type check
        self.check = CheckMixin

        # inits channel
        self.pubkey_channel = self.commu.guest2host_single_channel('public_key')
        self.encrypted_id_channel = self.commu.guest2host_single_channel('encrypted_id')
        self.double_encrypted_id_channel = self.commu.guest2host_single_channel('double_encrypted_id')
        self.intersection_id_channel = self.commu.guest2host_single_channel('intersection_id')

    @ClassMethodAutoLog()
    def _check_input(self, ids: List):
        # input data type check
        self.check.list_type_check(ids)


class RSASalGuest(RSASalBaseModel):
    """
        Federated data sharing: get aligned ids.
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        RSASalBaseModel.__init__(self,
                                 federal_info=federal_info,
                                 sec_param=sec_param,
                                 algo_param=algo_param)

    @ClassMethodAutoLog()
    def align(self, ids: List, *args, **kwargs) -> List:
        """
        Use RSA method to align the data from host and guest.

        Args:
            ids: list, data set

        Returns:
            data intersection
        -----

        **Example:**
        >>> ids = list(range(1000))
        >>> share = RSASalGuest(federal_info, sec_param, algo_param)
        >>> result = share.align(ids)
        """
        # check input data
        self._check_input(ids)
        self.logger.info('complete input data check')

        key_msg = self.save_first_pub_private_key()
        pub_key, pri_key = key_msg['pub_key'], key_msg["pri_key"]
        self.pubkey_channel.send(pub_key)
        self.logger.info('Guest has sent public key to host.')

        param = dict()
        param['rsa_n'] = pub_key.n
        param['rsa_d'] = pri_key.d

        # guest_ids_list = multi_process(func=_apply_guest_encrypt, data=ids, param=param)
        guest_ids_list = _apply_guest_encrypt(ids, param)
        self.logger.info('Guest has calculated the encrypted ids.')

        host_ids = self.encrypted_id_channel.recv()
        self.logger.info('Guest has received the host encrypted ids.')

        # host_ids_list = multi_process(func=_apply_guest_to_host_encrypt, data=host_ids, param=param)
        host_ids_list = _apply_guest_to_host_encrypt(host_ids, param)
        del host_ids
        gc.collect()
        self.logger.info('Guest encrypt the host ids by using private key.')

        self.double_encrypted_id_channel.send((host_ids_list, guest_ids_list))
        del host_ids_list, guest_ids_list
        gc.collect()
        self.logger.info('Guest has sent host and guest ids.')

        intersection_ids = self.intersection_id_channel.recv()
        self.logger.info('Guest has received aligned ids.')

        return intersection_ids.tolist()


class RSASalHost(RSASalBaseModel):
    """
        Secure alignment protocol based RSA method, host side.
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        RSASalBaseModel.__init__(self,
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

        pub_key = self.pubkey_channel.recv()
        self.logger.info('Host has received the public key from guest.')

        rand_num = random.SystemRandom().getrandbits(128)

        param = dict()
        param['rsa_n'] = pub_key.n
        param['rsa_e'] = pub_key.e
        param['rand_num'] = rand_num

        # ids_list = multi_process(func=_apply_host_encrypt, data=ids, param=param)
        ids_list = _apply_host_encrypt(ids, param)
        self.logger.info('Host has encrypted self ids.')
        self.encrypted_id_channel.send(ids_list)
        del ids_list
        gc.collect()
        self.logger.info('Host has sent self ids.')

        double_encrypted_ids = self.double_encrypted_id_channel.recv()
        self.logger.info("Host has receive guest's encrypted ids.")

        hashed_guest_ids = double_encrypted_ids[1]
        host_ids = double_encrypted_ids[0]
        del double_encrypted_ids
        gc.collect()

        # hashed_host_ids = multi_process(func=_apply_double_host_encrypt, data=host_ids, param=param)
        hashed_host_ids = _apply_double_host_encrypt(host_ids, param)
        self.logger.info("Host has encrypted self ids twice.")
        _, host_ids, _ = np.intersect1d(hashed_host_ids, hashed_guest_ids, return_indices=True)
        intersection_ids = np.array(ids)[host_ids]
        del hashed_guest_ids, host_ids, hashed_host_ids
        gc.collect()
        self.logger.info('Host get the aligned IDs.')
        self.intersection_id_channel.send(intersection_ids)
        self.logger.info('Host send aligned IDs.')

        return intersection_ids.tolist()


def _apply_guest_to_host_encrypt(data: List[Union[int, float, str]], param: Dict,
                                 *args, **kwargs) -> List:
    res = list(map(lambda x: powmod(x, param['rsa_d'], param['rsa_n']), data))
    return res


def _apply_guest_encrypt(data: List[Union[int, float, str]], param: Dict,
                         *args, **kwargs) -> List:
    res = list(map(lambda x: id_hash(powmod(id_hash(x), param['rsa_d'], param['rsa_n'])), data))
    return res


def _apply_host_encrypt(data: List[Union[int, float, str]], param: Dict,
                        *args, **kwargs) -> List:
    res = list(map(lambda x: mul(powmod(param['rand_num'], param['rsa_e'], param['rsa_n']),
                                 f_mod(id_hash(x), param['rsa_n'])), data))
    return res


def _apply_double_host_encrypt(data: List[Union[int, float, str]], param: Dict,
                               *args, **kwargs) -> List:
    res = list(map(lambda x: id_hash(divm(x, param['rand_num'], param['rsa_n'])), data))
    return res


def id_hash(align_id):
    return int(hashlib.sha256(str(align_id).encode('utf-8')).hexdigest(), 16)

