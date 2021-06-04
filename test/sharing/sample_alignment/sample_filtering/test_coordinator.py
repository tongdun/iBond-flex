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
from flex.constants import BF_SF

from test.fed_config_example import fed_conf_coordinator


def test_sample_filtering():
    # 联邦通信信息，输入
    federal_info = fed_conf_coordinator

    # 安全参数，输入
    sec_param = []

    # coordinator执行样本过滤协议，并将filter求交后的结果返回给双方
    make_protocol(BF_SF, federal_info, sec_param).intersect()
    print("done")
