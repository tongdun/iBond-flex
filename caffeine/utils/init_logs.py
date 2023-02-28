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

import logging
import logging.config
import os
import yaml
from os import path

from .config import log_dir

config_folder = path.abspath(path.dirname(__file__))
default_config = path.join(config_folder, 'log_setting.yaml')
log_level = os.getenv('LOG_LEVEL', logging.INFO)


def setup_logging(config_path: str = default_config) -> None:
    """
    Setup logging configuration
    """
    if path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        config["handlers"]["debug_file_handler"]["filename"] = config["handlers"]["debug_file_handler"][
            "filename"].replace(
            '${log_dir}', log_dir
        )
        config["handlers"]["info_file_handler"]["filename"] = config["handlers"]["info_file_handler"][
            "filename"].replace(
            '${log_dir}', log_dir
        )
        config["handlers"]["warning_file_handler"]["filename"] = config["handlers"]["warning_file_handler"][
            "filename"].replace(
            '${log_dir}', log_dir
        )
        config["handlers"]["error_file_handler"]["filename"] = config["handlers"]["error_file_handler"][
            "filename"].replace(
            '${log_dir}', log_dir
        )
        config["root"]["level"] = log_level
        logging.config.dictConfig(config)
    else:
        print('Found no logging config in %s, using default settings.' %
              config_path)
        logging.basicConfig(level=log_level)
