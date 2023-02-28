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

import numpy as np
import torch

from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor
from flex.crypto.paillier.keypair import generate_paillier_keypair
from flex.crypto.paillier import parallel_ops
from flex_test.utils import almost_equal


pe, pd = generate_paillier_encryptor_decryptor()


def test_generator():
    pk1, sk1 = generate_paillier_keypair(1024, seed=1)
    pk2, sk2 = generate_paillier_keypair(1024, seed=1)
    pk3, sk3 = generate_paillier_keypair(1024, seed=2)
    assert pk1.n.bit_length() == pk2.n.bit_length()
    assert pk1 == pk2 and pk1 != pk3


def test_encrypt_decrypt():
    # float
    plain_param1 = random.random()
    encrypted_param1 = pe.encrypt(plain_param1)
    plain_param2 = pd.decrypt(encrypted_param1)
    assert almost_equal(plain_param1, plain_param2)

    # one dim array
    plain_param1 = np.random.random(100).astype(np.float32)
    encrypted_param1 = pe.encrypt(plain_param1)
    plain_param2 = pd.decrypt(encrypted_param1)
    assert almost_equal(plain_param1, plain_param2)

    # two dim array
    plain_param1 = np.random.random((128, 4)).astype(np.float32)
    encrypted_param1 = pe.encrypt(plain_param1)
    plain_param2 = pd.decrypt(encrypted_param1)
    assert almost_equal(plain_param1, plain_param2)


def test_add():
    x1 = random.random()
    x2 = random.random()
    y1 = np.random.random(100).astype(np.float32)
    y2 = np.random.random(100).astype(np.float32)
    z1 = np.random.random((128, 4)).astype(np.float32)
    z2 = np.random.random((128, 4)).astype(np.float32)

    enc_x1 = pe.encrypt(x1)
    enc_x2 = pe.encrypt(x2)
    enc_y1 = pe.encrypt(y1)
    enc_y2 = pe.encrypt(y2)
    enc_z1 = pe.encrypt(z1)
    enc_z2 = pe.encrypt(z2)

    assert almost_equal(pd.decrypt(enc_x1 + enc_x2), x1 + x2)
    assert almost_equal(pd.decrypt(enc_y1 + enc_y2), y1 + y2)
    assert almost_equal(pd.decrypt(enc_z1 + enc_z2), z1 + z2)

    assert almost_equal(pd.decrypt(enc_x1 + x2), x1 + x2)
    assert almost_equal(pd.decrypt(enc_y1 + y2), y1 + y2)
    assert almost_equal(pd.decrypt(enc_z1 + z2), z1 + z2)


def test_mul():
    x1 = random.random()
    x2 = random.random()
    y1 = np.random.random(100).astype(np.float32)
    y2 = np.random.random(100).astype(np.float32)

    enc_x1 = pe.encrypt(x1)
    enc_y1 = pe.encrypt(y1)

    assert almost_equal(pd.decrypt(enc_x1 * x2), x1 * x2) # float * float
    assert almost_equal(pd.decrypt(enc_y1 * x1), y1 * x1) # float * array
    assert almost_equal(pd.decrypt(enc_y1 * y2), y1 * y2) # array * array


def test_add_mul_numpy_parallel():
    x = np.random.random(100).astype(np.float32)
    y = np.random.random(100).astype(np.float32)
    en_x = pe.encrypt(x)
    en_y = pe.encrypt(y)

    en_result = parallel_ops.add(en_x, y)
    result = pd.decrypt(en_result)
    assert almost_equal(x + y, result)

    en_result = parallel_ops.add(en_x, en_y)
    result = pd.decrypt(en_result)
    assert almost_equal(x + y, result)

    en_result = parallel_ops.mul(en_x, y)
    result = pd.decrypt(en_result)
    assert almost_equal(x * y, result)
