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
from flex.api import make_protocol
from flex.constants import OTP_STATS

from flex_test.fed_config_example import fed_conf_multiparty_guest1
from flex_test.utils import almost_equal

def test_otp_statistic():
    federal_info = fed_conf_multiparty_guest1

    sec_param = [['onetime_pad', {'key_length': 512}], ]

    algo_param = {}

    protocol = make_protocol(OTP_STATS, federal_info, sec_param, algo_param)

    data = np.random.rand(10000000)
    # data[[2, 3, 6]] = np.nan

    for _ in range(1):
        protocol.exchange(data, stats=['std'])
        assert almost_equal(protocol.data_sta['count'], data.size * 3)
        print('count', protocol.data_sta['count'])
        # print('not_null_count', protocol.data_sta['not_null_count'])
        print('mean', protocol.data_sta['mean'])
        assert almost_equal(protocol.data_sta['mean'], 0.5, 0.01)
        print('std', protocol.data_sta['std'])
        assert almost_equal(protocol.data_sta['std'], 0.28, 0.05)

        print('local_count', protocol.data_sta['local_count'])
        # print('local_not_nll_count', protocol.data_sta['local_not_null_count'])
        print('local_mean', protocol.data_sta['local_mean'])
        print('local_std', protocol.data_sta['local_std'])


if __name__ == '__main__':
    test_otp_statistic()
