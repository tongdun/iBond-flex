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

from flex.constants import HE_ML
from flex.api import make_protocol

from flex_test.fed_config_example import fed_conf_guest

def test(sz=100, cnt=10):
    federal_info = fed_conf_guest

    sec_param = [['paillier', {"key_length": 1024}], ]

    UID = 'user_A'
    REQVAL = 600

    protocol = make_protocol(HE_ML, federal_info, sec_param)
    
    for idx in range(cnt):
        result = protocol.exchange(UID, REQVAL)
        assert not result

if __name__ == '__main__':
    import sys
    nargs = len(sys.argv)
    if nargs == 1:
        test()
    elif nargs == 2:
        test(sz=int(sys.argv[1]))
    else:
        test(sz=int(sys.argv[1]), cnt=int(sys.argv[2]))