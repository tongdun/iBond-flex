from flex.api import make_protocol
from flex.constants import SAL

from flex_test.fed_config_example import fed_conf_coordinator


def test():
    federal_info = fed_conf_coordinator

    sec_param = [['aes', {'key_length': 128}]]

    algo_param = {}

    # align
    share = make_protocol(SAL, federal_info, sec_param, algo_param)
    share.align()

    # verify
    share.verify()


if __name__ == '__main__':
    test()
