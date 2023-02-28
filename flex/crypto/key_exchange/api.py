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
from typing import List

from .diffie_hellman import DiffieHellman


def make_agreement(remote_id: List, local_id: str, job_id: str , key_length: int = 2048) -> int:
    """
    N-party key exchange protocol api.

    Args:
        remote_id: list, remote ID which is configured in route table.
        local_id: str, local ID.
        key_length: int, key length, must in [2048, 3072, 4096, 6144, 8192]
    Returns:
        int, secret key

    -----

    **Example:**

    >>>remote_id = ["zhibang-d-011040", "zhibang-d-011041", "zhibang-d-011042"]
    >>>local_id = "zhibang-d-011040"
    >>>key_length = 2048
    >>>secret_key = make_agreement(remote_id=remote_id, local_id=local_id, key_length=key_length, job_id='hhh')
    """
    dh = DiffieHellman(key_length)
    secret_key = dh.key_exchange(remote_id, local_id, job_id)
    return secret_key
