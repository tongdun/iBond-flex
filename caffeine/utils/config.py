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
#  File name: config                                                                          
#                                                                                              
#  Create date: 2020/11/25                                                                               
#
import json
import os
import pathlib
from flex import constants

conf_path = os.getenv("CAFFEINE_CONF_PATH", None)
try:
    with open(conf_path, 'r') as config_file:
        config = json.load(config_file)
except:
    config = {}

version = config.get('version', '0.1')

audit_switch = config.get('audit_switch', False)

log_dir = os.getenv('CAFFEINE_LOG_PATH', None)
if log_dir is None:
    log_dir = config.get('log_dir', '.')
try:
    pathlib.Path.mkdir(pathlib.Path(log_dir), exist_ok=True)
except:
    log_dir = '.'

middleware_version = config.get(
    'middleware_version',
    'light'
)

security_config = {
    constants.HE_OTP_LR_FT1: [["paillier", {"key_length": 1024}]],
    constants.HE_OTP_LR_FT2: [["paillier", {"key_length": 1024}]],
    #constants.HE_LR_FP: [["paillier", {"key_length": 1024}]],
    #constants.HE_LR_FP: [["ckks", {"key_length": 8192}]], # TODO HE_LR_FP not support ckks
    constants.HE_LR_FP: None,
    constants.HE_LR_FP2: [["paillier", {"key_length": 1024}]],
    constants.OTP_SA_FT: [['onetime_pad', {'key_length': 512}]],
    # constants.OTP_PN_FL: None,
    constants.OTP_NN_FT: None,
    constants.HE_GB_FT: [["paillier", {"key_length": 1024}]],
    constants.HE_GB_FP: None,
    constants.SAL: [["aes", {"key_length": 128}]],
    constants.ECDH_SAL: None,
    constants.RSA_SAL: [['rsa', {"key_length": 2048}]],
    constants.OTP_PN_FL: [["onetime_pad", {"key_length": 512}]],
    constants.SS_STATS : [["secret_sharing", {"precision": 4}]],
    constants.FMC: [['secret_sharing', {"shift": 16, 'length': 64, 'p_value': 67}]]
}
file_security_config = config.get(
    'security_config',
    {}
)
security_config.update(file_security_config)


if __name__ == '__main__':
    import sys

    # write
    config = {}
    config['version'] = version
    config['audit_switch'] = audit_switch
    config['log_dir'] = log_dir
    config['middleware_version'] = middleware_version
    config['security_config'] = security_config

    with open(sys.argv[1], 'w') as config_file:
        json.dump(config, config_file, indent=4)
