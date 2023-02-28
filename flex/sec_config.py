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
"""
This file set same default parameters of crypto
"""
# Button for communication
ENABLE_ARROW = True

# support encrypt method
SEC_METHOD = ['aes',
              'sm4',
              'sm3',
              'rsa',
              'md5',
              'paillier',
              'secp256k1',
              'secret_sharing',
              'onetime_pad',
              'ot'
              ]


# default parameters for all encryption methods
SEC_DICT = {
    'rsa': {'key_length': 2048},
    'paillier': {'key_length': 1024},
    'aes': {'key_length': 128},
    'sm4': {'key_length': 128},
    'secp256k1': {'key_length': 256},
    'secret_sharing': {'precision': 3},
    'onetime_pad': {'key_length': 512},
    'sm3': {},
    'md5': {},
    'ot': {'n': 10, 'k': 1}
}


# encrypt method depend on diffie hellman
DEPEND_DH = ['aes', 'sm4', 'onetime_pad', 'ot']


# diffie hellman key length
DH_KEY_LENGTH = 2048


# proportion of free cores in parallel
DATA_LENGTH = 1000
FREE_CORE_RATIO = 0.1
FREE_MEMORY_RATIO = 0.20
DEFAULT_CORE = 1
DEFAULT_MEMORY_CORE = 5
