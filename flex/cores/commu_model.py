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

import copy
import os
from typing import Dict, List, Any, Optional

from flex.cores.parser import parse_federal_info
from flex.utils import ClassMethodAutoLog
from flex.sec_config import ENABLE_ARROW

if ENABLE_ARROW:
    try:
        from flex.commu.ionic_so import commu as commu

        commu.LocalTest = (os.getenv('COMMU_LOCALTEST') == 'TRUE')
        commu.UnitTest = (os.getenv('COMMU_UNITTEST') == 'TRUE')

        from flex.commu.ionic_so.channel import (
            make_broadcast_channel,
            make_variable_channel,
            # P2PNetworkChannel,
            VariableChannel,
            SignalChannel,
            create_channels)

    except ImportError:
        raise RuntimeError("Can not import ionic_bond, please install it and try again.")

else:
    print("Using flex.ionic_bond for communication.")
    import flex.ionic_bond.commu as commu

    commu.LocalTest = (os.getenv('COMMU_LOCALHOST') == 'TRUE')
    commu.UnitTest = (os.getenv('COMMU_UNITTEST') == 'TRUE')

    from flex.ionic_bond.channel import (
        make_broadcast_channel,
        make_variable_channel,
        VariableChannel)


def make_raw_channel(channel_name: str,
                     endpoint1: str, endpoint2: str,
                     job_id: Optional[str], *args, **kwargs) -> VariableChannel:
    """
    Construct channel between endpoint1 and endpoint2
    Arg:
        channel_name: str, name of channel
        endpoint1: str, ip value of endpoint1
        endpoint2: str, ip value of endpoint2
        job_id: str, unique representation of current job
    return:
        object
    """
    if ENABLE_ARROW:
        channel = make_variable_channel(channel_name,
                                        endpoint1,
                                        endpoint2,
                                        job_id=job_id)
    else:
        channel = make_variable_channel(channel_name,
                                        endpoint1,
                                        endpoint2)
    return channel


def make_raw_broadcast(channel_name: str,
                       root: str, remote_group: List[str],
                       job_id: Optional[str], *args, **kwargs) -> make_broadcast_channel:
    """
    Construct channels between root and remote group
    Arg:
        channel_name: str, name of channel
        root: str, ip value of root
        remote_group: list, ip values of remote group
        job_id: str, unique representation of current job
    return:
        object
    """
    if ENABLE_ARROW:
        channel = make_broadcast_channel(channel_name,
                                         root=root,
                                         remote_group=remote_group,
                                         job_id=job_id)
    else:
        channel = make_broadcast_channel(channel_name,
                                         root=root,
                                         remote_group=remote_group)
    return channel


class CommuModel:
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
        self._job_id = self.federal_info.job_id

        # communication inits, when SYNC_ON_START=false, communication initiation will not be checked by framework.
        sync_on_start = os.getenv('SYNC_ON_START', 'true')
        if sync_on_start == 'false':
            sync_on_start = False
        else:
            sync_on_start = True
        commu.init(federal_info, sync_on_start=sync_on_start)


    def guest2host_single_channel(self, channel_name: str) -> VariableChannel:
        """
        Construct channel between guest and host, guest/host has only one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_channel(channel_name,
                                   endpoint1=self.first_guest_id,
                                   endpoint2=self.first_host_id,
                                   job_id=self._job_id)
        return channel

    def guest2host_id_channel(self, host_party_id: str, channel_name: str) -> VariableChannel:
        """
        Construct channel from guest to specified host
        Arg:
            host_party_id: str, host id name
            channel_name: str, name of channel
        return:
            object
        """
        channel = make_raw_channel(channel_name,
                                   endpoint1=self.first_guest_id,
                                   endpoint2=host_party_id,
                                   job_id=self._job_id)
        return channel

    def guest2coord_single_channel(self, channel_name: str) -> VariableChannel:
        """
        Construct channel between guest and coord, guest/coord has only one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_channel(channel_name,
                                   endpoint1=self.first_guest_id,
                                   endpoint2=self.first_coord_id,
                                   job_id=self._job_id)
        return channel

    def host2coord_single_channel(self, channel_name: str) -> VariableChannel:
        """
        Construct channel between host and coord, host/coord has only one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_channel(channel_name,
                                   endpoint1=self.first_host_id,
                                   endpoint2=self.first_coord_id,
                                   job_id=self._job_id)
        return channel

    def guest2host_broadcast_channel(self, channel_name) -> make_broadcast_channel:
        """
        Guest construct channel to host, host more than one party
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_broadcast(channel_name,
                                     root=self.first_guest_id,
                                     remote_group=self.host_id,
                                     job_id=self._job_id)
        return channel

    def guest2guest_broadcast_channel(self, channel_name) -> make_broadcast_channel:
        """
        In cross sample, first guest construct channel to other guests
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_broadcast(channel_name,
                                     root=self.first_guest_id,
                                     remote_group=self.other_guest,
                                     job_id=self._job_id)
        return channel

    def coord_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel to guest and host
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_broadcast(channel_name,
                                     root=self.first_coord_id,
                                     remote_group=self.guest_host,
                                     job_id=self._job_id)
        return channel

    def coord2guest_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel only to guest
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_broadcast(channel_name,
                                     root=self.first_coord_id,
                                     remote_group=self.guest_id,
                                     job_id=self._job_id)
        return channel

    def coord2host_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel only to host
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_broadcast(channel_name,
                                     root=self.first_coord_id,
                                     remote_group=self.host_id,
                                     job_id=self._job_id)
        return channel

    def guest_broadcast_channel(self, channel_name: str) -> make_broadcast_channel:
        """
        Coordinator construct channel only to host
        Arg:
            channel_name: name of channel
        return:
            object
        """
        channel = make_raw_broadcast(channel_name,
                                     root=self.first_guest_id,
                                     remote_group=self.coord_host,
                                     job_id=self._job_id)
        return channel

    def party_broadcast(self, root: str, channel_name: str) -> make_broadcast_channel:
        """This method define communication channel when root channel is input, include coordinator"""
        return make_raw_broadcast(channel_name,
                                  root=root,
                                  remote_group=self.get_party_rest(root=root),
                                  job_id=self._job_id)

    def get_party_rest(self, root: str) -> List[str]:
        other = copy.deepcopy(self.guest_host + self.federal_info.coordinator)
        other.remove(root)
        return other

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
    def first_guest_host(self) -> str:
        return self.federal_info.guest_host[0]

    @property
    def second_guest_host(self) -> str:
        return self.federal_info.guest_host[1]

    @property
    def other_party(self) -> List:
        return self.federal_info.other_parties

    @property
    def local_id(self) -> str:
        return self.federal_info.local_id

    @property
    def all_party(self) -> List:
        return self.federal_info.all_party
