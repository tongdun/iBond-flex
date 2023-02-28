import time
from memory_profiler import profile
import numpy as np
from numpy.random import RandomState

from flex.api import make_protocol
from flex.constants import HE_FM_FP
from flex_test.fed_config_example import fed_conf_guest


@profile
def test_fm_fp():
    """
    Args:
    guest_theta: weight params of FM model's linear term
    guest_v: weight params of FM model's embedding term
    guest_features: origin dataset
    """
    federal_info = fed_conf_guest

    sec_param = []

    prng = RandomState(0)
    batch_size = 32
    feature_num = 1000
    guest_theta = prng.uniform(-1, 1, (feature_num, ))
    guest_v = prng.uniform(-1, 1, (feature_num, 10))
    guest_features = prng.uniform(-1, 1, (batch_size, feature_num))

    start = time.time()
    trainer = make_protocol(HE_FM_FP, federal_info, sec_param, algo_param=None)

    predict = trainer.exchange(guest_theta, guest_v, guest_features)
    end = time.time()
    print(predict)
    print(f'time is {end - start}')


if __name__ == '__main__':
    test_fm_fp()
