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

import logging
import logging.config
from typing import Tuple, Dict, Optional
from os import path
import yaml
from .config import log_dir

config_folder = path.abspath(path.dirname(__file__))
default_config = path.join(config_folder, 'log_setting.yaml')


def setup_logging(config_path: str = default_config,
                  default_level: str = logging.INFO) -> None:
    """
    Setup logging configuration
    """
    if path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        config["handlers"]["info_file_handler"]["filename"] = config["handlers"]["info_file_handler"]["filename"].replace(
            '${log_dir}', log_dir
        )
        config["handlers"]["error_file_handler"]["filename"] = config["handlers"]["error_file_handler"]["filename"].replace(
            '${log_dir}', log_dir
        )
        logging.config.dictConfig(config)
    else:
        print('Found no logging config in %s, using default settings.' %
              config_path)
        logging.basicConfig(level=default_level)
