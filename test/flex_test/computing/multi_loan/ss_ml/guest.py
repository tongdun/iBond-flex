"""ss_ml of multi_loan protocol: guest
"""
from flex.constants import SS_ML
from flex.api import make_protocol

from flex_test.fed_config_example import fed_conf_guest


def test():
    federal_info = fed_conf_guest

    sec_param = [['paillier', {'key_length': 1024}], ]

    algo_param = {
    }
    UID = 'user_A'
    REQVAL = 600

    protocol = make_protocol(SS_ML,
                             federal_info,
                             sec_param,
                             algo_param)
    result = protocol.exchange(UID, REQVAL)  # pylint: disable-all
    print(result)


if __name__ == '__main__':
    test()
