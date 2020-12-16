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

from .keypair import PaillierPublicKey
from .. import gmpy_math
from .obfuscator import apply_obfuscation


def raw_encrypt(plaintext: int, pub_key: PaillierPublicKey, random_value: int = None) -> int:
    """
    Encode a int into a encrypted number. This method is also used by encrypted_number.
    Args:
        plaintext: int or float, as input number.
        pub_key: a PaillierPublicKey
        random_value: int, used to do obfuscate if given.

    Returns: 
        int, encrypt number.
    """
    if not isinstance(plaintext, int):
        raise TypeError("plaintext should be int, but got: %s" %
                        type(plaintext))

    if pub_key.n > plaintext >= (pub_key.n - pub_key.max_int):
        # Very large plaintext, take a sneaky shortcut using inverses
        neg_plaintext = pub_key.n - plaintext
        neg_ciphertext = (gmpy_math.mul(
            pub_key.n, neg_plaintext) + 1) % pub_key.nsquare
        ciphertext = gmpy_math.invert(neg_ciphertext, pub_key.nsquare)
    else:
        ciphertext = (gmpy_math.mul(pub_key.n,
                                    plaintext) + 1) % pub_key.nsquare

    ciphertext = apply_obfuscation(ciphertext, pub_key, random_value)

    return ciphertext
