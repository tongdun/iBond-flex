"""ss_ml: host
"""
from flex.constants import SS_ML
from flex.api import make_protocol

from flex_test.fed_config_example import fed_conf_host


def test():
    federal_info = fed_conf_host

    sec_param = [['paillier', {'key_length': 1024}], ]

    algo_param = {
    }

    def req_loan(uid):
        return 500.0

    protocol = make_protocol(SS_ML,
                             federal_info,
                             sec_param,
                             algo_param)
    protocol.exchange(req_loan)


if __name__ == '__main__':
    test()
