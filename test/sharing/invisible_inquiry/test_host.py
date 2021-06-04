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
from flex.constants import OT_INV

from test.fed_config_example import fed_conf_host


def test_invisible_inquiry():
    fed_conf_host['session']['identity'] = 'server'
    federal_info = fed_conf_host

    sec_param = [['aes', {'key_length': 128}], ['ot', {'n': 10, 'k': 1}]]

    algo_param = {}

    protocol = make_protocol(OT_INV, federal_info, sec_param, algo_param)

    # 模拟的匿踪查询函数
    def query_fun(in_list):
        result = [str(int(i) * 100) for i in in_list]
        return result

    # 连续做10次匿踪查询，并将结果返回给client端
    for i in range(10):
        protocol.exchange(query_fun)
