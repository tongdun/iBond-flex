"""dot_mul: coordinator
"""
from flex.api import make_protocol
from flex.constants import SS_COMPUTE

from test.fed_config_example import fed_conf_coordinator


def test():
    federal_info = fed_conf_coordinator
    sec_param = [['secret_sharing', {'precision': 3}], ]
    algo_param = {
    }

    protocol = make_protocol(SS_COMPUTE, federal_info, sec_param, algo_param)

    protocol.share_secrets(None)
    protocol.substract_rec()
    protocol.le_rec()


if __name__ == '__main__':
    test()
