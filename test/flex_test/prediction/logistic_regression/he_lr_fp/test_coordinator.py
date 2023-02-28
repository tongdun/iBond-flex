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

from flex.api import make_protocol
from flex.constants import HE_LR_FP
from flex_test.fed_config_example import fed_conf_coordinator


def test_predict():
    len_u1 = 2

    federal_info = fed_conf_coordinator

    sec_param = [['paillier', {"key_length": 1024}], ]

    protocol = make_protocol(HE_LR_FP, federal_info, sec_param, algo_param=None)

    for i in range(len_u1):
        protocol.exchange()

