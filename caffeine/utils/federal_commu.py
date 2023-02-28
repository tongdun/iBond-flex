#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: federal_info_parser.py
#
#  Create date: 2021/01/26
#
from flex.commu.ionic_so import commu as commu
from flex.commu.ionic_so.channel import make_broadcast_channel
from typing import Optional, Dict, List

from caffeine.utils.exceptions import ParseError


class FederationParser(object):
    """
    Parse federation info to attributes.
    """
    def __init__(self, federation: Dict[str, List[str]]):
        self._federation = federation

    @property
    def federation(self):
        return self._federation

    @property
    def guests(self) -> List:
        return sorted(self._federation.get('guest', []))

    @property
    def major_guest(self) -> str:
        if len(self.guests) <= 0:
            raise ParseError(f'There is no guest in federation.')
        return self.guests[0]

    @property
    def hosts(self) -> List:
        return sorted(self._federation.get('host', []))

    @property
    def major_host(self) -> str:
        if len(self.hosts) <= 0:
            raise ParseError(f'There is no host in federation.')
        return self.hosts[0]

    @property
    def coords(self) -> List:
        return sorted(self._federation.get('coordinator', []))

    @property
    def major_coord(self):
        if len(self.coords) <= 0:
            raise ParseError(f'There is no coordinator in federation.')
        return self.coords[0]

    @property
    def participants(self):
        return self.guests + self.hosts

    @property
    def all(self):
        return self.coords + self.guests + self.hosts


class FederalInfoParser(FederationParser):
    def __init__(self, federal_info: Dict):
        """
        Parse federal info to attributes.
        """
        self._federal_info = federal_info
        self._federation = self._federal_info['federation']
        self._server = self._federal_info['server']
        self._session = self._federal_info['session']

    @property
    def federal_info(self):
        return self._federal_info

    @property
    def server(self):
        return self._server

    @property
    def job_id(self):
        return self._session['job_id']

    @property
    def local_id(self):
        return self._session['local_id']

    @property
    def local_role(self):
        return self._session['role']


class IonicUser(FederationParser):
    def __init__(self, 
                 channel_names_types: Dict[str, tuple], 
                 federation: Dict[str, List[str]],
                 job_id: Optional[str] = None
                ):
        """
        Init ionic channels. Channels will be inserted into self __dict__ as
            f'_{channel_name}_chan'.

        Args:
            channel_names_types: dict of string, keys are channel names and
                values are tuples (channel_type, *args), the channel types are in ['broadcast'].
            federation: dict of string, federation in federal info
            job_id: optional string, specified job_id

        -----

        **Examples:**

        >>> user = IonicUser()
        >>> federation = {
                'coordinator': ['c01', 'c02'],
                'guest': ['g01', 'g02', 'g03'],
                'host': []
            }
        >>> user.init_channels({'sync': ('broadcast', 'c01')}, federation)
        >>> user.coord_id
        'c01'
        """
        super().__init__(federation)

        for name, value in channel_names_types.items():
            channel_type = value[0]
            if channel_type == 'broadcast':
                root_id = value[1]
                remote_ids = [i for i in self.all if i != root_id]
                self.__dict__[f'_{name}_chan'] = make_broadcast_channel(
                    name = name,
                    root = root_id,
                    remote_group = remote_ids,
                    job_id = job_id
                )


class Radio(IonicUser):
    def __init__(self, 
                 station_id: str, 
                 federal_info: Dict, 
                 channels: List[str] = ['default']):
        """
        Initiate the radio.

        Args:
            station_id: str, the party id of the radio station.
            federal_info: the federal information, including jobid, federation, etc..
            channels: a list of strings, specifying the channel names.
        """
        commu.init(federal_info)
        self._station_id = station_id
        self._channels = channels
        channel_names_types = {
            name: ('broadcast', self._station_id) for name in self._channels
        }
        super().__init__(
            channel_names_types = channel_names_types,
            federation = federal_info.get('federation'),
            job_id = federal_info.get('session', {}).get('job_id')
        )

    @property 
    def station_id(self):
        return self._station_id

    @property
    def channels(self):
        return self._channels


if __name__ == '__main__':
    import uuid
    from multiprocessing import Process

    ################################
    job_uuid = uuid.uuid1().hex

    federation = {
        'guest': ['g01'], 
        'host': ['h01'], 
        'coordinator': ['c01'], 
    }
    station_id = 'g01'
    ################################

    expanded = [(r, i) for r in federation for i in federation[r]]

    fed_configs = {i: {
        "server": "localhost:6149",
        "session": {
            "role": r,
            "local_id": i,
            "job_id": job_uuid
        },
        "federation": federation
    } for r, i in expanded}

    def run(i):
        ratio = Radio(
            station_id, 
            fed_configs[i],
            ['test_radio', 'test_ratio_1']
        )
        if i == station_id:
            ratio._test_radio_chan.broadcast(f"Hello, I'm station {i}.")
            msgs = ratio._test_radio_chan.gather()
            print(f'Station {i} received: {msgs}')
        else:
            msg = ratio._test_radio_chan.broadcast()
            print(f'Audience {i} received: {msg}')
            ratio._test_radio_chan.gather(f"Hello, I'm audience {i}.")

    process_list = []
    for i in fed_configs:
        p = Process(target=run, kwargs={'i': i})
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
