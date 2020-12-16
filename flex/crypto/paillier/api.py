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

from .keypair import generate_paillier_keypair, PaillierPrivateKey, PaillierPublicKey
from .encryptor import PaillierEncryptor
from .decryptor import PaillierDecryptor


def generate_paillier_encryptor_decryptor(n_length: int = 1024, seed: int = None) -> (PaillierEncryptor, PaillierDecryptor):
    public_key, private_key = generate_paillier_keypair(n_length, seed)
    return PaillierEncryptor(public_key), PaillierDecryptor(public_key, private_key)


def generate_paillier_encryptor(n: int) -> PaillierEncryptor:
    return PaillierEncryptor(PaillierPublicKey(n))


def generate_paillier_decryptor(n: int, p: int, q: int) -> PaillierDecryptor:
    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)
    return PaillierDecryptor(public_key, private_key)
