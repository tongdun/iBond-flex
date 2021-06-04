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
from typing import Optional, Dict, List

from flex.cores.parser import parse_federal_info
from flex.utils import ClassMethodAutoLog
from flex.sec_config import ENABLE_ARROW

if ENABLE_ARROW:
    try:
        import ionic_bond.commu as commu

        commu.LocalTest = (os.getenv('COMMU_LOCALTEST') == 'TRUE')
        commu.UnitTest = (os.getenv('COMMU_UNITTEST') == 'TRUE')

        from ionic_bond.channel import make_broadcast_channel, make_variable_channel, \
            VariableChannel, SignalChannel, create_channels

    except ImportError:
        raise RuntimeError("Can not import ionic_bond, please install it and try again.")

else:
    print("Using flex.ionic_bond for communication.")
    import flex.ionic_bond.commu as commu

    commu.LocalTest = (os.getenv('COMMU_LOCALHOST') == 'TRUE')
    commu.UnitTest = (os.getenv('COMMU_UNITTEST') == 'TRUE')

    from flex.ionic_bond.channel import make_broadcast_channel, make_variable_channel, \
        VariableChannel, SignalChannel, create_channels       


class CommuModel:
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict):
        """
        inits of federation information fro communication
        Args:
            federal_info: dict, federal info

        Example:
        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "host",
        >>>        "local_id": "zhibang-d-014010",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> CommuModel(federal_info)
        """
        self.federal_info = parse_federal_info(federal_info)

        # communication inits
        commu.init(federal_info)

    def make_raw_channel(self, channel_name: str, endpoint1: str, endpoint2: str) -> VariableChannel:
        """
        Construct channel between endpoint1 and endpoint2
        Arg:
            channel_name: name of channel
            endpoint1: endpoint1
            endpoint2: endpoint2
        return:
            object
        """
        channel = make_variable_channel(channel_name,
                                        endpoint1,
                                        endpoint2)
        return channel

    @ClassMethodAutoLog()
    def guest2host_single_channel(self, channel_name: str) -> VariableChannel:
        """
        Construct channel between guest and host, guest/host has only one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_variable_channel(channel_name,
                                        endpoint1=self.first_guest_id,
                                        endpoint2=self.first_host_id)
        return channel

    @ClassMethodAutoLog()
    def guest2host_id_channel(self, host_party_id: str, channel_name: str) -> VariableChannel:
        """
        Construct channel from guest to specified host
        Arg:
            host_party_id: str, host id name
            channel_name: str, name of channel
        return:
            object
        """
        channel = make_variable_channel(channel_name,
                                        endpoint1=self.first_guest_id,
                                        endpoint2=host_party_id)
        return channel

    @ClassMethodAutoLog()
    def guest2coord_single_channel(self, channel_name: str) -> VariableChannel:
        """
        Construct channel between guest and coord, guest/coord has only one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_variable_channel(channel_name,
                                        endpoint1=self.first_guest_id,
                                        endpoint2=self.first_coord_id)
        return channel

    @ClassMethodAutoLog()
    def host2coord_single_channel(self, channel_name: str) -> VariableChannel:
        """
        Construct channel between host and coord, host/coord has only one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_variable_channel(channel_name,
                                        endpoint1=self.first_host_id,
                                        endpoint2=self.first_coord_id)
        return channel

    @ClassMethodAutoLog()
    def guest2host_broadcast_channel(self, channel_name) -> make_broadcast_channel:
        """
        Guest construct channel to host, host more than one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_broadcast_channel(channel_name,
                                         root=self.first_guest_id,
                                         remote_group=self.host_id)
        return channel

    @ClassMethodAutoLog()
    def guest2guest_broadcast_channel(self, channel_name) -> make_broadcast_channel:
        """
        In cross sample, first guest construct channel to other guests
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_broadcast_channel(channel_name,
                                         root=self.first_guest_id,
                                         remote_group=self.other_guest)
        return channel

    @ClassMethodAutoLog()
    def coord_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel to guest and host
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_broadcast_channel(channel_name,
                                         root=self.first_coord_id,
                                         remote_group=self.guest_host)
        return channel

    @ClassMethodAutoLog()
    def coord2guest_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel only to guest
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_broadcast_channel(channel_name,
                                         root=self.first_coord_id,
                                         remote_group=self.guest_id)
        return channel

    @ClassMethodAutoLog()
    def coord2host_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel only to host
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_broadcast_channel(channel_name,
                                         root=self.first_coord_id,
                                         remote_group=self.host_id)
        return channel

    @ClassMethodAutoLog()
    def guest_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel only to host
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_broadcast_channel(channel_name,
                                         root=self.first_guest_id,
                                         remote_group=self.coord_host)
        return channel

    @property
    def first_host_id(self) -> str:
        return self.federal_info.host[0]

    @property
    def first_guest_id(self) -> str:
        return self.federal_info.guest[0]

    @property
    def first_coord_id(self) -> str:
        return self.federal_info.coordinator[0]

    @property
    def other_guest(self) -> List:
        return self.federal_info.guest[1:]

    @property
    def host_id(self) -> List:
        return self.federal_info.host

    @property
    def guest_id(self) -> List:
        return self.federal_info.guest

    @property
    def coord_host(self) -> List:
        return self.federal_info.coord_host

    @property
    def guest_host(self) -> List:
        return self.federal_info.guest_host

    @property
    def local_id(self) -> str:
        return self.federal_info.local_id
