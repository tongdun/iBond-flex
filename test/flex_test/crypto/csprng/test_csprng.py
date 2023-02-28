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


from flex.crypto.csprng.api import generate_csprng_generator
from flex.constants import *


def test_drbg():
    entropy = b'542435464554342576476747656736767657676545234546'
    personalization_string = b''
    num_bytes = 100

    output1 = b'>\xb3N\x89Y*\xa6NF\xeff\xe1\xcb\xec\\\x90\x19\xf1\xb6 \xc3 \xcd~\xc4BO\x83\xd5}\xfaAkk\x8f\xbd\x8f\x80\x168\x8f[\x86\\\xd8\xd3\x03\x91\xbbR\xe5\x9c<\xa2 \x11}\xf8\xc9^\xc3\xef(\x9a\x99\xb9e\xa3\xa9\x1c\x1a$\x9b\xb1\x19k;*-&\xaa\x05\n%\xdf\xdf\xf3\x08\xdc\xdb\xf6\x86\xe8\xa8\x15\xb6\xfc\xff\xb6\xd6'
    output2 = b'Ggf\x08|\xc6\xb7\x7f\xd7\xc4\x80\xf5\xee\xb6"\xe4\xd0w\xa8\xe032\xd9%\xee|C\xf2\xd1\xd3n\xdb\xa3\xad\xc5:&k\x95\x13)\xc6\x91\xb8\xe7\xb7\x0fI\x0c\xeb\xf8g\xbfEj]\xa7\x0e\x9a\xe1O\xa0\x7fd\x10~=\x0bN,\xb2\xda1\x99{\xc5\xd7\xcde\xa3\xed\x9c\x15\x1f8\x9f\xd6\xd5~b\x91\x86\xa0P\xf5\xcb\x9d\xa4x\xcc\x9f\xc7\x99\xca\x87\xfa+\xcbu\x02x2\x12\xba6P\x19\x86\xe6\xb2\xfa\xb4\t"\xdb\xb5\x19z'

    drbg = generate_csprng_generator(entropy, personalization_string, method=CRYPTO_HMAC_DRBG)
    onetime_key1 = drbg.generate(num_bytes)
    assert onetime_key1 == output1

    drbg.reseed(b'e4243546455434c576476747656736767657676d4523454a')
    onetime_key2 = drbg.generate(2**7)
    assert onetime_key2 == output2
