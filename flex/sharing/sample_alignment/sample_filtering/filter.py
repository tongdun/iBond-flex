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
import time
import math
import hashlib
from typing import Dict, List, Optional, Union
from functools import reduce

import numpy as np

from flex.crypto.id_filter.api import generate_id_filter
from flex.crypto.csprng.api import generate_csprng_generator
from flex.crypto.id_filter.id_filter import IDFilter
from flex.utils import ClassMethodAutoLog
from flex.cores.base_model import BaseModel
from flex.sec_config import DH_KEY_LENGTH


class BFSFBaseModel(BaseModel):
    """
    Apply secure intersection protocol of id_filters.
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        Sample filtering protocol param inits
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

        >>> sec_param = []

        >>> algo_param = {"log2_len": 31}

        >>> BFSFParty(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        # inits communication
        self.filter_channel = self.commu.coord_broadcast_channel('filter')

        self.remote_id = None
        self.local_id = None

    def _gen_random_filter(self, seed: Union[int, bytes], log2_len: int) -> IDFilter:
        """
        Generate a IDFilter by a random seed.

        Args:
            seed: int or bytes, used for generate presudo random number.
            log2_len: log 2 of filter length.

        Returns:
            An IDFilter instance.
        """
        # use Deterministic Random Bit Generators to generate a seed to gen Random Filter
        bit_length = (1 << log2_len)
        hmac_drbg = generate_csprng_generator(seed, b"", method="hmac_drbg")

        random_filter = np.zeros(math.ceil(bit_length / 8), dtype=np.uint8)

        num_bytes = 640
        start = 0
        end = num_bytes

        for i in range(math.ceil((1 << log2_len) / (8 * num_bytes)) - 1):
            random_bytes = hmac_drbg.generate(num_bytes=num_bytes)
            random_filter[start: end] = np.frombuffer(random_bytes, dtype=np.uint8)
            start = end
            end += num_bytes

        num_bytes = len(random_filter) - start
        random_bytes = hmac_drbg.generate(num_bytes=num_bytes)
        random_filter[start: end] = np.frombuffer(random_bytes, dtype=np.uint8)

        random_filter = generate_id_filter(log2_len, random_filter)

        return random_filter

    def _intersect(self, id_filter: Union[IDFilter, None]) -> Union[IDFilter, None]:
        """
        Apply secure intersection of id_filters.

        Args:
            id_filter: IDFilter generated from ID list.

        Returns:
            Intersection of id_filters for non-coordinator roles, None for coordinator.
        """
        # use DH key exchange to get seed for IDFilter
        key_permute = self._generate_diffie_hellman_seed(self.remote_id, self.local_id, DH_KEY_LENGTH)
        time.sleep(1)
        key_mask = self._generate_diffie_hellman_seed(self.remote_id, self.local_id, DH_KEY_LENGTH)
        self.logger.info('use DH key exchange to get seed for IDFilter')

        # convert the seed to bytes
        key_permute = key_permute.to_bytes(math.ceil(key_permute.bit_length() / 8), 'big')
        key_mask = key_mask.to_bytes(math.ceil(key_mask.bit_length() / 8), 'big')
        self.logger.info('convert the seed to bytes')

        # hash to bit length 256, prepare for FF1 encryption
        key_permute = hashlib.sha256(key_permute).digest()
        key_mask = hashlib.sha512(key_mask).digest()
        self.logger.info('hash to bit length 256, prepare for FF1 encryption')

        # permute the id filter
        permuted_id_filter = id_filter.permute(key_permute)

        # mask the id filter
        mask_filter = self._gen_random_filter(key_mask, id_filter.log2_bit_length)
        self.logger.info('permute and mask the id filter')

        # send the filter to coord to do intersection and receive intersection
        encrypted_filter = permuted_id_filter == mask_filter
        self.filter_channel.gather(encrypted_filter)
        intersected_encrypted_filter = self.filter_channel.broadcast()
        self.logger.info('send the filter to coord to do intersect and receive intersection')

        # decrypt the intersected filter in local
        intersected_filter = intersected_encrypted_filter & permuted_id_filter
        intersected_filter = intersected_filter.inv_permute(key_permute)
        self.logger.info('decrypt the intersected filter in local')

        return intersected_filter


class BFSFCoord(BFSFBaseModel):
    """
    Sample filtering protocol, Coordinator side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        BFSFBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

    def intersect(self, *args, **kwargs) -> None:
        """
        Receive the IDFilter from host and guest, than calculation the intersection and send it back

        Returns:
            None
        """
        # calculation the intersection
        encrypted_filter_list = self.filter_channel.gather()
        filter_list_head = encrypted_filter_list[0]
        filter_list_tail = encrypted_filter_list[1:]
        intersected_encrypted_filter = [filter_list_head == x for x in filter_list_tail]
        intersected_encrypted_filter = reduce(lambda x, y: x & y, intersected_encrypted_filter)
        self.logger.info('Complete calculation of the intersection')

        # coordinator send the intersection to every party
        self.filter_channel.broadcast(intersected_encrypted_filter)
        self.logger.info('coordinator send the intersection to every party')


class BFSFParty(BFSFBaseModel):
    """
    Sample filtering protocol, non-coordinator side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        BFSFBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

        # get the remote id and local id to do key exchange
        self.remote_id = self.commu.guest_host
        self.local_id = self.commu.local_id

    def intersect(self, id_filter: Union[IDFilter, None],
                  *args, **kwargs) -> Union[IDFilter, None]:
        """
        Apply secure intersection of id_filters.

        Args:
            id_filter: IDFilter generated from ID list.

        Returns:
            Intersection of id_filters.
        """
        return self._intersect(id_filter)
