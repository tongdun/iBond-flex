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

from flex.api import make_protocol
from flex.constants import OT_INV
from flex_test.fed_config_example import fed_conf_guest


def test_invisible_inquiry():
    # 联邦通信信息，输入，根据当前的配置环境做相应修改
    fed_conf_guest['session']['identity'] = 'client'
    federal_info = fed_conf_guest

    # 安全参数，输入
    sec_param = [['aes', {'key_length': 128}], ['ot', {'n': 10, 'k': 1}]]

    # 算法参数，输入
    algo_param = {}

    protocol = make_protocol(OT_INV, federal_info, sec_param, algo_param)

    import random

    # 模拟的混淆函数，用于生成和查询id相同格式的混淆id
    def obfuscator(in_list, n):
        fake_list = [random.randint(0, 100) for i in range(n - len(in_list))]
        index = random.randint(0, n - 1)
        joint_list = fake_list[:index] + in_list + fake_list[index:]
        return joint_list, index

    # 模拟的匿踪查询函数
    def query_fun(in_list):
        result = [str(int(i) * 100) for i in in_list]
        return result

    # 输入的查询id从1到10做10次不同测试，将调用匿踪查询后server返回的结果和本地计算结果进行比较，验证正确性
    for i in range(10):
        federal_result = protocol.exchange(str(i), obfuscator)  # 联邦匿踪查询结果
        local_result = query_fun([str(i)])[0]  # 本地查询结果，查询函数同联邦匿踪查询函数
        assert federal_result == local_result
