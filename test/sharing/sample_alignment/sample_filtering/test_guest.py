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
from flex.crypto.id_filter.id_filter import IDFilter
from flex.constants import BF_SF

from test.fed_config_example import fed_conf_guest
from flex.cores.ionic import commu
from flex.cores.ionic import make_variable_channel


def test_sample_filtering():
    # 联邦通信信息，输入，根据当前的配置环境做相应修改
    federal_info = fed_conf_guest

    # 通信初始化，创建通信信道
    commu.init(federal_info)
    var_chan = make_variable_channel('test_data_exchange', federal_info["federation"]["guest"][0],
                                     federal_info["federation"]["host"][0])
    # 安全参数，输入
    sec_param = []

    # 算法参数，输入
    algo_param = {
        "log2_len": 23
    }

    # guest端的测试数据，输入
    input_data = list(range(6000, 14000))

    # 将测试数据加载到长度为2^log2_len长度的bloom filter中，同时为了减少通信开销，将filter长度压缩为原先的1/8
    fltr = IDFilter(algo_param['log2_len'])
    fltr.update(input_data)

    # 将测试数据发送给求交的另一参与方host，并接收从另一方发送的数据，由于求交是在压缩的filter上进行的，求完交集后需要将对齐数据解压
    recv_filter = var_chan.swap(fltr.filter, tag="filter")
    intersected_filter = fltr.filter & recv_filter
    original_filter = np.unpackbits(intersected_filter)

    # 联邦求双方filter的交集，并将求交集后的filter解压
    interset_fltr = make_protocol(BF_SF, federal_info, sec_param)
    sf_res = interset_fltr.intersect(fltr)
    inter_fltr = np.unpackbits(sf_res.filter)

    # 断言联邦返回的filter和本地计算的filter是否相同
    assert np.alltrue(original_filter == inter_fltr)
