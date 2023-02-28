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

from flex.crypto.onetime_pad.api import generate_onetime_pad_encryptor
from flex.crypto.onetime_pad.iterative_add import iterative_add
from flex_test.utils import almost_equal


def test_encrypt_decrypt():
    secret_key = b'1' * 48
    encryptor = generate_onetime_pad_encryptor(secret_key)
    x = np.random.uniform(-1, 1, (512,)).astype(np.float32)
    en_x = encryptor.encrypt(x, alpha=1)
    z = encryptor.decrypt(en_x, alpha=1)
    assert almost_equal(z, x)

    x = torch.Tensor(512, ).uniform_(-1, 1)
    en_x = encryptor.encrypt(x, alpha=1)
    z = encryptor.decrypt(en_x, alpha=1)
    assert almost_equal(z, x)

    x = [[np.random.uniform(-1, 1, (512,)).astype(np.float32)], [torch.Tensor(512, ).uniform_(-1, 1)]]
    en_x = encryptor.encrypt(x, alpha=1)
    z = encryptor.decrypt(en_x, alpha=1)
    assert almost_equal(z, x)


def test_add_diff():
    secret_key = b'1' * 48
    encryptor1 = generate_onetime_pad_encryptor(secret_key)
    encryptor2 = generate_onetime_pad_encryptor(secret_key)

    x = np.array([random.random()], dtype=np.float32)
    y = np.array([random.random()], dtype=np.float32)
    enc_x = encryptor1.encrypt(x, alpha=-1)
    enc_y = encryptor2.encrypt(y, alpha=2)
    assert almost_equal(encryptor1.decrypt(enc_x + enc_y, alpha=1), x + y)

    x = np.random.uniform(-1, 1, (512,)).astype(np.float32)
    y = np.random.uniform(-1, 1, (512,)).astype(np.float32)
    enc_x = encryptor1.encrypt(x, alpha=-1)
    enc_y = encryptor2.encrypt(y, alpha=2)
    assert almost_equal(encryptor1.decrypt(enc_x + enc_y, alpha=1), (x + y))

    x = torch.Tensor(512, ).uniform_(-1, 1)
    y = torch.Tensor(512, ).uniform_(-1, 1)
    enc_x = encryptor1.encrypt(x, alpha=-1)
    enc_y = encryptor2.encrypt(y, alpha=2)
    assert almost_equal(encryptor1.decrypt(enc_x + enc_y, alpha=1), (x + y))

    x = [[np.random.uniform(-1, 1, (512,)).astype(np.float32)], [torch.Tensor(512, ).uniform_(-1, 1)]]
    y = [[np.random.uniform(-1, 1, (512,)).astype(np.float32)], [torch.Tensor(512, ).uniform_(-1, 1)]]
    enc_x = encryptor1.encrypt(x, alpha=-1)
    enc_y = encryptor2.encrypt(y, alpha=2)
    assert almost_equal(encryptor1.decrypt(enc_x + enc_y, alpha=1), iterative_add(x, y))


def test_add_decode():
    secret_key = b'1' * 48
    encryptor1 = generate_onetime_pad_encryptor(secret_key)
    encryptor2 = generate_onetime_pad_encryptor(secret_key)

    x = [[np.random.uniform(-1, 1, (512,)).astype(np.float32)], [torch.Tensor(512, ).uniform_(-1, 1)]]
    y = [[np.random.uniform(-1, 1, (512,)).astype(np.float32)], [torch.Tensor(512, ).uniform_(-1, 1)]]
    enc_x = encryptor1.encrypt(x, alpha=-1)
    enc_y = encryptor2.encrypt(y, alpha=1)
    assert almost_equal((enc_x + enc_y).decode(), iterative_add(x, y))
