"""ss_ml multi_loan: coordinator
"""
from flex.constants import SS_ML
from flex.api import make_protocol

from flex_test.fed_config_example import fed_conf_coordinator

def test():
    federal_info = fed_conf_coordinator

    sec_param = [['paillier', {'key_length': 1024}], ]

    algo_param = {
    }

    protocol = make_protocol(SS_ML, federal_info, sec_param, algo_param)
    protocol.exchange()


if __name__ == '__main__':
    test()
