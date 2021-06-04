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

"""ss_compute: host
"""
import torch

from flex.api import make_protocol
from flex.constants import SS_COMPUTE

from test.fed_config_example import fed_conf_host


def test(sz=100, cnt=10):
    federal_info = fed_conf_host

    sec_param = [['secret_sharing', {'precision': 0}], ]

    algo_param = {
    }

    torch.manual_seed(seed=1111111)
    B = torch.randint(100, [sz, sz])
    protocol = make_protocol(SS_COMPUTE, federal_info, sec_param, algo_param)

    ret = protocol.share_secrets(B)
    a_sh, b_sh = ret

    for idx in range(cnt):

        ret = protocol.ge_rec(a_sh, b_sh)
        print(ret)


if __name__ == '__main__':
    import sys
    nargs = len(sys.argv)
    if nargs == 1:
        test()
    elif nargs == 2:
        test(sz=int(sys.argv[1]))
    else:
        test(sz=int(sys.argv[1]), cnt=int(sys.argv[2]))
