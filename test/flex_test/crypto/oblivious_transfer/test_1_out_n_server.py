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


from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.cores.commu_model import commu
from flex_test.fed_config_example import fed_conf_guest


def test_ot():
    commu.init(fed_conf_guest)

    ot_protocol = make_ot_protocol(1, 10, 'zhibang-d-014010')
    for times in range(10):
        server_msg = [str(i) for i in range(10)]
        ot_protocol.server(server_msg)
