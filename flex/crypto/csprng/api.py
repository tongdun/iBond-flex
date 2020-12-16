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
from typing import Union

from .hmac_drbg import HMacDRBG
from flex.constants import *


def generate_csprng_generator(entropy: Union[int, bytes], personalization_string: bytes = b"",
                              method=CRYPTO_HMAC_DRBG) -> CRYPTO_HMAC_DRBG:
    if isinstance(entropy, int):
        length = math.ceil(entropy.bit_length() / 8)
        entropy_bytes = entropy.to_bytes(length, 'big')
    elif isinstance(entropy, bytes):
        entropy_bytes = entropy
    else:
        raise TypeError(f"Entropy type {type(entropy)} is not valid.")

    if method == CRYPTO_HMAC_DRBG:
        return HMacDRBG(entropy_bytes, personalization_string)
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
