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
from flex.constants import OTP_STATS

from test.fed_config_example import fed_conf_multiparty_coordinator

def test_otp_statistic():
    federal_info = fed_conf_multiparty_coordinator

    sec_param = [['onetime_pad', {'key_length': 512}], ]

    algo_param = {}

    protocol = make_protocol(OTP_STATS, federal_info, sec_param, algo_param)
    for _ in range(1):
        protocol.exchange(stats=['std'])


if __name__ == '__main__':
    test_otp_statistic()
