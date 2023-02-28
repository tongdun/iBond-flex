#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name: mixins.py                                                                          
#                                                                                              
#  Create date: 2020/06/03                                                                             
#
from typing import List, Dict, Union, Optional
import math

from pydantic import BaseModel
from flex.api import make_protocol
from flex import constants
from flex.commu.ionic_so.channel import make_broadcast_channel, make_variable_channel
from flex.commu.ionic_so import commu

from caffeine.utils import ClassMethodAutoLog
from caffeine.utils.exceptions import DataMismatchError, UninitializedError
from caffeine.utils.common_tools import gen_module_id
from caffeine.model.base_model import JsonModule, JsonModel
from caffeine.feature.config import security_config


class FLEXUser(object):
    def init_protocols(self, protocols: Dict):
        """
        Init FLEX protocols.

        Args:
            protocols, a list of protocol names.
        """
        self._protocols = dict()
        for p, v in protocols.items():
            if constants.FMC in p:
                pn = p.split('_')[-1]
            else:
                pn = p
            self._protocols[p] = make_protocol(
                                    pn,
                                    self._meta_params['federal_info'],
                                    self._meta_params['security_param'].get(p),
                                    v
                                ) 

    def init_protocols_2party(self, protocols: Dict, federal_info: Dict):
        self._protocols_2party = {
            p: make_protocol(
                p,
                federal_info,
                self._meta_params['security_param'].get(p),
                v
            ) for p, v in protocols.items()
        }

class FederationParser(object):
    """
    Parse federation info to attributes.
    """
    @property
    def _federation(self):
        return self._meta_params['federal_info']['federation']

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
    @property
    def federal_info(self):
        return self._meta_params['federal_info']

    @property
    def server(self):
        self._server = self.federal_info['server']
        return self._server

    @property
    def _session(self):
        return self.federal_info['session']

    @property
    def job_id(self):
        return self._session['job_id']

    @property
    def local_id(self):
        return self._session['local_id']

    @property
    def local_role(self):
        return self._session['role']

class IonicUser(FederalInfoParser):
    def init(self):
        commu.init(self.federal_info)

    def init_channels(self, channel_names_types: Dict, root_id: str=None, remote_group: list=None):
        """
        Init ionic channels. Channels will be inserted into self __dict__ as
            f'_{channel_name}_chan'.

        Args:
            channel_names_types, dict, keys are channel names and values are the
                channel types in ['broadcast', 'exchange'].
        """
        self.init()
        if root_id is None:
            for name in channel_names_types:
                if channel_names_types[name] == 'broadcast':
                    self.__dict__[f'_{name}_broadcast'] = make_broadcast_channel(
                        name=name,
                        root=self.major_guest,
                        remote_group=self.hosts + self.coords,
                        job_id=self.job_id
                    )
                if channel_names_types[name] == 'exchange':
                    self.__dict__[f'_{name}_chan'] = make_broadcast_channel(
                        name=name,
                        root=self.major_guest,
                        remote_group=self.hosts,
                        job_id=self.job_id
                    )
        else:
            for name in channel_names_types:
                if channel_names_types[name] == 'broadcast':
                    self.__dict__[f'_{name}_broadcast'] = make_broadcast_channel(
                        name=name,
                        root=root_id,
                        remote_group=remote_group,
                        job_id=self.job_id
                    )
                if channel_names_types[name] == 'exchange':
                    self.__dict__[f'_{name}_chan'] = make_broadcast_channel(
                        name=name,
                        root=root_id,
                        remote_group=remote_group,
                        job_id=self.job_id
                    )

