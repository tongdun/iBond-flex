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


# 验证自造数据的结果正确性

import numpy as np
from flex.api import make_protocol
from flex.constants import SAL

from flex_test.fed_config_example import fed_conf_guest


def test_secure_alignment():
    federal_info = fed_conf_guest

    sec_param = [['aes', {'key_length': 128}]]

    algo_param = {}

    data_1 = [list(map(str, range(1000, 2000))), list(range(3000, 5000))]
    data_2 = [list(map(str, range(1600))), list(range(4200))]

    # align
    share = make_protocol(SAL,  federal_info, sec_param, algo_param)
    for i, data in enumerate(data_1):
        result = share.align(data)
        _, idx_1, _ = np.intersect1d(data_1[i], data_2[i], return_indices=True)
        local_res = [data[j] for j in idx_1]
        assert sorted(result) == sorted(local_res)

        # verify
        is_align = share.verify(result)
        assert is_align is True
