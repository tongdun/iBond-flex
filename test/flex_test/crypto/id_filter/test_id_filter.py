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
from os import urandom

import pytest
import numpy as np

from flex.crypto.id_filter.api import generate_id_filter


def test_init_id_filter():
    log2_len = [8, 17, 24, 21]
    src_filter = [
        np.random.randint(0, 2, (2**8,)).astype(np.bool),
        np.random.randint(0, 256, (2**17 // 8,)).astype(np.uint8),
        np.random.randint(0, 256, (2**24 // 8 + 1)).astype(np.uint8),
        np.random.randint(0, 255, (2**21 // 8)).astype(np.float32)
    ]

    for i in range(2):
        generate_id_filter(log2_len[i], src_filter[i])

    with pytest.raises(ValueError):
        generate_id_filter(log2_len[2], src_filter[2])

    with pytest.raises(TypeError):
        generate_id_filter(log2_len[3], src_filter[3])


def test_permute():
    for i in range(10):
        log2_len = random.randint(7, 22)
        src_filter = np.random.randint(0, 256, (2**log2_len // 8,)).astype(np.uint8)
        secret_key = urandom(random.choice([16, 24, 32]))

        filter1 = generate_id_filter(log2_len, src_filter)
        filter2 = filter1.permute(secret_key)
        filter3 = filter2.inv_permute(secret_key)

        assert np.all(filter1.filter == filter3.filter)


def test_equal_op():
    for i in range(10):
        log2_len = random.randint(7, 22)
        src_filter1 = np.random.randint(0, 256, (2 ** log2_len // 8,)).astype(np.uint8)
        src_filter2 = np.random.randint(0, 256, (2 ** log2_len // 8,)).astype(np.uint8)

        filter1 = generate_id_filter(log2_len, src_filter1)
        filter2 = generate_id_filter(log2_len, src_filter2)

        expected_output = np.packbits(np.unpackbits(src_filter1) == np.unpackbits(src_filter2))

        assert np.all((filter1 == filter2).filter == expected_output)


def test_and_op():
    for i in range(10):
        log2_len = random.randint(7, 22)
        src_filter1 = np.random.randint(0, 256, (2 ** log2_len // 8,)).astype(np.uint8)
        src_filter2 = np.random.randint(0, 256, (2 ** log2_len // 8,)).astype(np.uint8)

        filter1 = generate_id_filter(log2_len, src_filter1)
        filter2 = generate_id_filter(log2_len, src_filter2)

        expected_output = np.packbits(np.unpackbits(src_filter1) & np.unpackbits(src_filter2))

        assert np.all((filter1 & filter2).filter == expected_output)


