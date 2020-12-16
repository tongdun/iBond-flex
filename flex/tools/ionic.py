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

import os
INDUSTRIAL = False

if not INDUSTRIAL:
    import flex.ionic_bond.commu as commu
    if os.getenv('COMMU_LOCALTEST') == 'TRUE':
        commu.LocalTest = True
    elif os.getenv('COMMU_UNITTEST') == 'TRUE':
        commu.UnitTest = True
    from flex.ionic_bond.channel import make_broadcast_channel, make_variable_channel, VariableChannel, SignalChannel, \
        create_channels
else:
    import ionic_bond.commu as commu
    if os.getenv('COMMU_LOCALTEST') == 'TRUE':
        commu.LocalTest = True
    elif os.getenv('COMMU_UNITTEST') == 'TRUE':
        commu.UnitTest = True
    from ionic_bond.channel import make_broadcast_channel, make_variable_channel, VariableChannel, SignalChannel, \
        create_channels
