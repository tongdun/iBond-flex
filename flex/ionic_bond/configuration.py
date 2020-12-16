"""Configurations for flex bond
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
import os
import logging
import sys
from typing import Dict
import tempfile


class Configuration():
    """Configuration loader
    """
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s;%(levelname)s;%(message)s',
                        level=logging.DEBUG)
    config = None
    session = None
    role = None
    local_id = None
    job_id = None
    federation = None
    federation_members = None

    socket_port = 16001
    socket_head = 4 * 1024
    socket_prefix = '/dev/shm'
    if not os.path.exists(socket_prefix):
        socket_prefix = os.path.join(tempfile.gettempdir(), 'msg_server')
    if not os.path.exists(socket_prefix):
        os.mkdir(socket_prefix)

    def __init__(self, config: Dict):
        """
        A client to communicate with flex bond server.

        Args:
            config: dict, from config.json


        Return:
            Ion instance.
        Example:
        >>> config = {
                        "server": "localhost:16001",
                        "session": {
                            "role": "host",
                            "local_id": "zhibang-d-014010",
                            "job_id": 'test_job',
                        },
                        "federation": {
                            "host":    ["zhibang-d-014010"],
                            "guest":   ["zhibang-d-014011"],
                            "arbiter": ["zhibang-d-014012"]
                        }
                    }
        >>> Ion(config)
        """
        # self.server_url = self.config.get("server")
        Configuration.config = config
        Configuration.federation = config.get("federation")
        Configuration.federation_members = [
            i for v in Configuration.federation.values() for i in v]

        if "session" not in config:
            config["session"] = {}
            config["session"]["local_id"] = platform.node()
            # in this case, job_id should in outer
            config["session"]["job_id"] = config.get("job_id")
            config["session"]["role"] = ""
            for _role, _list in config.get("federation").items():
                if config["session"]["local_id"] in _list:
                    config["session"]["role"] = _role

        session = config.get("session")
        Configuration.local_id = session.get("local_id")
        Configuration.role = session.get("role")
        Configuration.job_id = session.get("job_id")

    @staticmethod
    def get_config():
        """
        return config
        """
        return Configuration.config

    @staticmethod
    def get_hostname():
        """
        return hostname
        """
        return platform.node()


if __name__ == '__main__':
    import json
    with open('conf.example.json', 'r') as conf_file:
        conf = Configuration(json.load(conf_file))
    print('config:', conf.get_config())
