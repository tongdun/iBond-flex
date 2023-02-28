
import numpy as np

from flex.constants import HE_LR_FP2
from flex.api import make_protocol
from flex_test.fed_config_example import fed_conf_host


def test():
    u = np.random.uniform(-1, 1, (32,))
    print(u)

    federal_info = fed_conf_host

    sec_param = [['paillier', {"key_length": 1024}], ]

    predict = make_protocol(HE_LR_FP2, federal_info, sec_param, algo_param=None)
    result = predict.exchange(u)
    print(result)


if __name__ == '__main__':
    test()
