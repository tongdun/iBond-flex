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
from flex.constants import HE_SA_FT
from flex.api import make_protocol
from flex.tools.ionic import make_variable_channel
from test.fed_config_example import fed_conf_host
from flex.crypto.onetime_pad.iterative_add import iterative_add
from flex.tools.iterative_apply import iterative_divide
from test.utils import almost_equal


def test():
    theta = [
        [np.random.uniform(-1, 1, (2, 4)).astype(np.float32), np.random.uniform(-1, 1, (2, 6)).astype(np.float32)],
        [np.random.uniform(-1, 1, (2, 8)).astype(np.float32)]]
    print(theta)

    federal_info = fed_conf_host

    sec_param = {
        "he_algo": "paillier",
        "he_key_length": 1024,
        "key_exchange_size": 2048
    }

    trainer = make_protocol(HE_SA_FT, federal_info, sec_param)
    # 协议计算的平均梯度
    result = trainer.exchange(theta)
    var_chan = make_variable_channel('test_he_sa_ft', fed_conf_host["federation"]["host"][0],
                                     fed_conf_host["federation"]["guest"][0])
    guest_theta = var_chan.recv(tag='theta')
    sum_theta = iterative_add(theta, guest_theta)
    # 本地计算的平均梯度
    avg_theta = iterative_divide(sum_theta, 2.0)
    assert almost_equal(result, avg_theta)
