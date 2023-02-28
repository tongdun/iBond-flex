
from flex.constants import HE_LR_FP
from flex.api import make_protocol
from flex_test.fed_config_example import fed_conf_coordinator


def test():
    federal_info = fed_conf_coordinator

    sec_param = [['paillier', {"key_length": 1024}], ]

    predict = make_protocol(HE_LR_FP, federal_info, sec_param, algo_param=None)
    result = predict.exchange()
    print(result)


if __name__ == '__main__':
    test()
