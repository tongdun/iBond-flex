"""
Flex bond message channels
"""
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

from enum import Enum
from typing import Any, Union, List, Dict, Callable
from . import commu


class VariableChannel():
    """
    A VariableChannel can be used to communicate between local and
    remote[single/group] machines.
    It can be treated as a duel direction channel.
    """

    def __init__(self,
                 name: str,
                 remote_id: str,
                 auto_offset: bool = True):
        """
        Args:
            name: str, Base name for variable
            remote_id: str, Remote ID which is configed in route table.
            auto_offset: bool, Whether to use a automatic increment offset
            in full variable name.
        """
        self.check(remote_id)
        self.name: str = name
        self.remote_id: str = remote_id
        self.local_id: str = commu.get_local_id()
        self.job_id: str = commu.get_job_id()
        self.auto_offset: bool = auto_offset
        self.send_offset: int = 0
        self.recv_offset: int = 0

    @staticmethod
    def check(remote_id):
        """
        Check init parms
        """
        if commu.get_local_id() == remote_id:
            print(
                f"Warning, remote_id={remote_id} should be diferent with"+
                f" local_id={commu.get_local_id()}. ")
        if remote_id not in commu.get_federation_members():
            raise ValueError(
                f"Remote_id={remote_id} is not in federation,"
                " check your config!")

    def __send_name(self, tag: str = '*') -> str:
        """
        Define the full name of Variable to be send.
        """
        return f"{self.job_id}.{self.name}.{self.local_id}->{self.remote_id}" \
               f".offset={self.send_offset}.tag={tag}"

    def __recv_name(self, tag: str = '*') -> str:
        """
        Define the full name of Variable to be receive.
        """
        return f"{self.job_id}.{self.name}.{self.remote_id}->{self.local_id}" \
               f".offset={self.recv_offset}.tag={tag}"

    def send(self, var: Any, tag: str = '*') -> None:
        """
        Send local variable.

        Args:
            var: Any, Local variable to be sent.
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from remote endpoint.

        Example:
        >>> var_chan = VariableChannel(name="Exchange_secret_key",
                                       remote=remote_id)
        >>> var_chan.send(MyVar)
        """
        commu.send(value=var,
                   key=self.__send_name(tag),
                   dst=self.remote_id)

        if self.auto_offset:
            self.send_offset += 1

    def recv(self, tag: str = '*') -> Any:
        """
        Get remote variable.

        Args:
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from remote endpoint.

        Example:
        >>> var_chan = VariableChannel(name="Exchange_secret_key",
                                       remote=remote_id)
        >>> RemoteVar = var_chan.recv()
        """
        result = commu.recv(
            key=self.__recv_name(tag),
            src=self.remote_id)

        if self.auto_offset:
            self.recv_offset += 1

        return result

    def swap(self, var: Any, tag: str = '*') -> Any:
        """
        Swap local and remote variable.

        Args:
            var: Any, Local variable to be sent.
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from remote endpoint.

        Example:
        >>> var_chan = VariableChannel(name="Exchange_secret_key",
                                       remote=remote_id)
        >>> RemoteVar = var_chan.swap(MyVar)
        """
        self.send(var, tag)
        return self.recv(tag)

    def __enter__(self) -> "__class__":
        """
        A context mannager, which can make sure all remote member
        syncs before and after.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Just do another sync.
        """
        return

def make_variable_channel(name: str, endpoint1: str, endpoint2: str, auto_offset: bool = True) -> VariableChannel:
    """
    Args:
        name: str, Base name for variable
        endpoint1: str, ID for one endpoint in communication pair.
        endpoint2: str, ID for the other endpoint in communication pair.
        auto_offset: bool, Whether to use a automatic increment offset in full variable name.
    """
    def check(endpoint1, endpoint2):
        """
        Check init parms
        """
        for endpoint in (endpoint1, endpoint1):
            if endpoint not in commu.get_federation_members():
                raise ValueError(
                    f"Endpoint1={endpoint} is not in federation, check your config!")

        if endpoint1 == endpoint2:
            raise ValueError("Endpoint1 and endpoint2 should not be the same.")

        # if endpoint1 != commu.get_local_id() and endpoint2 != commu.get_local_id():
        #    raise ValueError(
        #        f"local_id={commu.get_local_id()} is neither endpoint1 or endpoint2, check your config!")

    check(endpoint1, endpoint2)

    if commu.get_local_id() == endpoint1:
        return VariableChannel(name, remote_id=endpoint2, auto_offset=auto_offset)
    elif commu.get_local_id() == endpoint2:
        return VariableChannel(name, remote_id=endpoint1, auto_offset=auto_offset)
    else:
        # This is used for symmetry.
        return None


class RemoteVariableBroadcastChannel():
    """
    BroadcastChannel is used to build a broad cast like way of communication.
    """

    def __init__(self,
                 name: str,
                 root: str,
                 remote_group: List[str],
                 auto_offset: bool = True):
        """
        Args:
            name: str, Base name for variable
            root: str, ID for the root process,
                  which is configed in route table
            remote_group: List[str], List of remote ID
                          which is configed in route table.
            auto_offset: bool, Whether to use a automatic increment offset
                         in full variable name.
        """
        self.root = root
        self.group = remote_group
        self.root_channel = VariableChannel(
            "Broadcast_"+name, self.root, auto_offset)

    def size(self):
        """remote group size
        """
        return len(self.group)

    def scatter(self, tag: str = '*') -> Any:
        """
        Receive root's local variable to all members of remote group [each with
        one item from original list].

        Args:
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from root [for remote group member].
            None for root.

        Example:
        In root code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> var_chan.scatter(['A', 'B'])

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> MyVar = var_chan.scatter() # If 'remote_id1' then MyVar == 'A'
        """
        return self.root_channel.recv(tag)

    def broadcast(self, tag: str = '*') -> Any:
        """
        Send root's local variable to all members of remote group.

        Args:
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from root [for remote group member].
            None for root.

        Example:
        In root code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> var_chan.broadcast(MyVar) # Send var to remote_group members.

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> MyVar = var_chan.broadcast() # Receive var from root.
        """
        return self.root_channel.recv(tag)

    def gather(self, var: Any, tag: str = '*') -> Any:
        """
        Get data from all remote group members into root's local variable.

        Args:
            var: Any, Local variable to be sent [only remote group].
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from remote group member [for root].
            None for remote group member.

        Example:
        In root code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> MyVar = var_chan.gather() # Send var to remote_group members.

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> var_chan.gather(MyVar) # Receive var from root.
        """
        self.root_channel.send(var, tag)

    def map(self, var: Any,  tag: str = '*') -> Any:
        """
        This map function applies a function in root to every item of iterable[from remote group], and get back the results. 
        Args:
            var: Any, Local variable to be sent [only remote group].
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            None

        Example:
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> var_chan.map(lambda x: x+1 )

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> MyVar_plus1 = var_chan.map(MyVar) 
        """
        self.gather(var, tag=tag)
        return self.scatter(tag=tag)

    def allreduce(self, var: Any,  tag: str = '*') -> Any:
        """
        This reduce function applies a reduce function in root for all items from iterable[from remote group], and broadcast back the results. 
        Args:
            var: Any, Local variable to be sent [only remote group].
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            None

        Example:
        >>> var_chan = make_broadcast_channel(name="Example", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> var_chan.allreduce(lambda x: sum(x))

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Example", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> MyVar_sum = var_chan.allreduce(MyVar) 
        """
        self.gather(var, tag=tag)
        return self.broadcast(tag=tag)

class RootVariableBroadcastChannel():
    """
    BroadcastChannel is used to build a broad cast like way of communication.
    """

    def __init__(self,
                 name: str,
                 root: str,
                 remote_group: List[str],
                 auto_offset: bool = True):
        """
        Args:
            name: str, Base name for variable
            root: str, ID for the root process,
                  which is configed in route table
            remote_group: List[str], List of remote ID
                          which is configed in route table.
            auto_offset: bool, Whether to use a automatic increment offset
                         in full variable name.
        """
        self.root = root
        self.group = remote_group
        self.my_channels = {
            remote_id: VariableChannel(
                "Broadcast_"+name,
                remote_id,
                auto_offset)
            for remote_id in self.group}

    def size(self):
        """remote group size
        """
        return len(self.group)

    def scatter(self, variables: List[Any], tag: str = '*') -> Any:
        """
        Send root's local variable to all members of remote group
            [each with one item from original list].

        Args:
            variables: List[Any], A group of items needs to be sent.
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from root [for remote group member].
            None for root.

        Example:
        In root code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> var_chan.scatter(['A', 'B'])

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> MyVar = var_chan.scatter() # If 'remote_id1' then MyVar == 'A'
        """
        if len(variables) != len(self.group):
            raise ValueError("Input variables must have same"
                             " length with remote group.")

        for var, chann_name in zip(variables, self.group):
            self.my_channels[chann_name].send(var, tag)

    def broadcast(self, var: Any, tag: str = '*') -> Any:
        """
        Send root's local variable to all members of remote group.

        Args:
            var: Any, Local variable to be sent [only for root].
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from root [for remote group member].
            None for root.

        Example:
        In root code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> var_chan.broadcast(MyVar) # Send var to remote_group members.

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> MyVar = var_chan.broadcast() # Receive var from root.
        """
        for remote_id in self.group:
            self.my_channels[remote_id].send(var, tag)

    def gather(self, tag: str = '*') -> Any:
        """
        Get data from all remote group members into root's local variable.

        Args:
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            Any, Receive variable from remote group member [for root].
            None for remote group member.

        Example:
        In root code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        # Get list of var from remote_group members.
        >>> VarList = var_chan.gather()

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key",
                                              root='local_id',
                                              remote_group=['remote_id1','remote_id2'])
        >>> var_chan.gather(MyVar) # Send var to root.
        """
        return [self.my_channels[remote_id].recv(tag)
                for remote_id in self.group]

    def map(self, func: Callable,  tag: str = '*') -> None:
        """
        This map function applies a function to every item of iterable[from remote group], and send back the results. 
        Args:
            func: Callable, Function or class method.
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            None

        Example:
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> var_chan.map(lambda x: x+1 )

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Exchange_secret_key", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> MyVar_plus1 = var_chan.map(MyVar) 
        """
        result = map(func, self.gather(tag=tag))
        self.scatter(result, tag=tag)

    def allreduce(self, func: Callable,  tag: str = '*') -> None:
        """
        This reduce function applies a reduce function in root for all items from iterable[from remote group], and broadcast back the results. 
        Args:
            var: Any, Local variable to be sent [only remote group].
            tag: str, Optional, if you want to custmize your variable tag.
        Return:
            None

        Example:
        >>> var_chan = make_broadcast_channel(name="Example", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> var_chan.allreduce(lambda x: sum(x))

        In remote group code.
        >>> var_chan = make_broadcast_channel(name="Example", root='local_id', remote_group=['remote_id1','remote_id2'])
        >>> MyVar_sum = var_chan.allreduce(MyVar) 
        """
        result = func(self.gather(tag=tag))
        self.broadcast(result, tag=tag)


def make_broadcast_channel(name: str,
                           root: str,
                           remote_group: List[str],
                           auto_offset: bool = True) \
        -> Union[RootVariableBroadcastChannel, RemoteVariableBroadcastChannel]:
    """
    Args:
        name: str, Base name for variable
        root: str, ID for the root process, which is configed in route table
        remote_group: List[str], List of remote ID
                      which is configed in route table.
        auto_offset: bool, Whether to use a automatic increment offset
                      in full variable name.
    """

    def check(root: str, remote_group: List[str]) -> None:
        """
        Check init parms
        """
        if root not in commu.get_federation_members():
            raise ValueError(
                f"Root={root} is not in federation {commu.get_federation_members()}, check your config!")

        for remote_id in remote_group:
            if remote_id not in commu.get_federation_members():
                raise ValueError(
                    f"Remote_id={remote_id} is not in federation,"
                    " check your config!")

        if root in remote_group:
            raise ValueError("Root id should not be included in remote_group")

        if root != commu.get_local_id() and commu.get_local_id() \
                not in remote_group:
            raise ValueError(
                f"local_id={commu.get_local_id()} is neither root"
                " or in the remote_group.")

    check(root, remote_group)

    if root == commu.get_local_id():
        return RootVariableBroadcastChannel(
            name, root, remote_group, auto_offset)
    return RemoteVariableBroadcastChannel(
        name, root, remote_group, auto_offset)


class Signal(Enum):
    """syncing with Signal
    """
    Sync = 1
    Stop = 2


class SignalChannel():
    """
    A SignalChannel can be used to communicate between local and
        remote[single/group] machines.
    It can be treated as a synchronizer.
    """

    def __init__(self, name: str, remote_id: str):
        """
        Args:
            name: str, Base name for variable
            remote_id: str, Remote ID which is configed in route table.
            auto_offset: bool, Whether to use a automatic increment offset
                         in full variable name.

        Example:
        >>> sc = SignalChannel(name="Sync", remote=remote_id)
        >>> sc.sync()
        Or using a Context manager style like this.
        >>> with SignalChannel(name="Sync", remote=remote_id) as sc:
        >>>     # sync is automaticly called once before do().
        >>>     do()
        >>>     # sync is automaticly called once after do().
        """
        self.chann = VariableChannel("Signal_"+name, remote_id, False)

    def sync(self) -> None:
        """
        Get synchronized with remote.
        """
        self.chann.swap(Signal.Sync)
        self.chann.swap(Signal.Sync)

    def __enter__(self) -> VariableChannel:
        """
        A context mannager, which can make sure all remote member
            syncs before and after.
        """
        self.sync()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Just do another sync.
        """
        self.sync()


def create_channels(config: Dict) -> Dict:
    """
    Generate a group of channels, using predefined config dict.
    Args:
        config: Dict,

    Return:
        Dict,  contains all channels with
            key->ChannelName, value->ChannelObject.

    Example:
        config = {
            "TestVar1":{
                "type":"VariableChannel",
                "remote_id": "zhibang-d-014011"
            },
            "TestVar2":{
                "type":"VariableChannel",
                "remote_id": "zhibang-d-014011"
            },
            "BTestVar1":{
                "type":"VariableBroadcastChannel",
                "root": "zhibang-d-014010",
                "remote_group": ["zhibang-d-014011","zhibang-d-014012"]
            }
        }
    """
    if not isinstance(config, dict):
        raise ValueError("Wrong type of config for make_channels.")

    channels = {}
    for key, value in config.items():
        if value.get("type") == "VariableChannel":
            channels[key] = VariableChannel(
                name=key,
                remote_id=value.get("remote_id"),
                auto_offset=value.get("auto_offset", True))
        elif value.get("type") == "VariableBroadcastChannel":
            channels[key] = make_broadcast_channel(
                name=key,
                root=value.get("root"),
                remote_group=value.get("remote_group"),
                auto_offset=value.get("auto_offset", True))
        else:
            print(f"Warning, channel type {key} is not supported.")

    return channels
