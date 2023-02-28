"""Configurations for MPC
   mpc uses fixed precision tensor to represent float types
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

import platform
import socket
import configparser
import logging
import sys

import torch

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s;%(levelname)s;%(message)s',
                    level=logging.DEBUG)


class Configuration():  # pylint: disable-all
    """MPC Configurations class
    """
    def __init__(self, jobid, conf_file, ibond_conf):
        self.jobid = jobid
        if isinstance(conf_file, str):
            self.config = configparser.ConfigParser()
            self.config.read(conf_file)
        else:
            self.config = conf_file  # dict
        self.hostname = platform.node()
        self.hostname_simple = self.hostname.split('.')[0]
        self.num_party = int(self.config['GENERAL']['NUM_PARTY'])
        if 'PRECISION' in self.config['GENERAL']:
            self.precision = int(self.config['GENERAL']['PRECISION'])
        else:
            self.precision = 0
        self.field = 2**64
        self.ip_addr = socket.gethostbyname_ex(self.hostname)[2][0]
        self.cuda = True if torch.cuda.is_available() else False

        self.addressbook = {}
        self.ibond_conf = ibond_conf
        self.wid2ibond = {}
        for _wid in range(self.num_party + 1):
            # item: hostname, role  # , ip
            _name, _role = self.config['ADDRESSBOOK'][str(_wid)].split(', ')
            self.addressbook[_wid] = _name, _role
            if _name in [self.hostname, self.hostname_simple, self.ip_addr]:
                self.role = _role
                self.world_id = _wid
            self.wid2ibond[_wid] = _name
        if 'session' in ibond_conf and 'local_id' in ibond_conf['session']:
            if 'role' in ibond_conf['session']:
                _role = ibond_conf['session']['role']
                self.role = 'ARBITER' if _role == 'coordinator' else 'PARTY'
            _flatted = ibond_conf['federation']['host']
            if 'guest' in ibond_conf['federation']:
                _flatted = _flatted + ibond_conf['federation']['guest']
            if 'coordinator' in ibond_conf['federation']:
                _flatted = _flatted + ibond_conf['federation']['coordinator']
            self.world_id = _flatted.index(ibond_conf['session']['local_id'])
