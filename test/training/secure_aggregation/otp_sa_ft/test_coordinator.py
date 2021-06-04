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
from flex.constants import OTP_SA_FT
from test.fed_config_example import fed_conf_coordinator


def test():
    federal_info = fed_conf_coordinator

    sec_param = [('onetime_pad', {"key_length": 512})]

    trainer = make_protocol(OTP_SA_FT, federal_info, sec_param)
    result = trainer.exchange()
