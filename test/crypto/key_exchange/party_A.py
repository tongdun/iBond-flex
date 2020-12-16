import os
import json

from flex.tools.ionic import commu
from flex.crypto.key_exchange.api import make_agreement
from test.fed_config_example import fed_conf_host


if __name__ == '__main__':
    commu.init(fed_conf_host)

    k = make_agreement(remote_id='zhibang-d-014011', key_size=2048)
    print(k)

