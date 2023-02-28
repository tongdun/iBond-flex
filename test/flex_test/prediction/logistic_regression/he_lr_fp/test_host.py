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


import numpy as np
import torch

from flex.api import make_protocol
from flex.constants import HE_LR_FP
from flex_test.fed_config_example import fed_conf_host


def test_predict():
    # u1 = [np.array([0.1, 0.05, -3.6, 25.8], dtype=np.float32), np.array([-0.5, 11.2, 9.5], dtype=np.float32)]
    u2 = [np.array([0.3, -14, -2.5, 1.7], dtype=np.float32), np.array([0.2, 1.2, -5.6], dtype=np.float32)]

    # expected_u = [u1[i] + u2[i] for i in range(len(u1))]

    federal_info = fed_conf_host

    sec_param = [['paillier', {"key_length": 1024}], ]

    protocol = make_protocol(HE_LR_FP, federal_info, sec_param, algo_param=None)

    for i in range(len(u2)):
        protocol.exchange(u2[i])
