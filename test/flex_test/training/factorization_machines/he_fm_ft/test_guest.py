import time
import numpy as np
from numpy.random import RandomState

from flex.api import make_protocol
from flex.constants import HE_FM_FT
from flex_test.fed_config_example import fed_conf_guest


def test_fm_ft():
    """
    Args:
    guest_theta: weight params of FM model's linear term
    guest_v: weight params of FM model's embedding term
    guest_features: origin dataset
    guest_labels: labels msg of guest
    """
    federal_info = fed_conf_guest

    sec_param = []

    prng = RandomState(0)
    batch_size = 32
    feature_num = 1000
    guest_theta = prng.uniform(-1, 1, (feature_num, ))
    guest_v = prng.uniform(-1, 1, (feature_num, 10))
    guest_features = prng.uniform(-1, 1, (batch_size, feature_num))
    guest_labels = prng.randint(0, 2, (batch_size, ))

    start = time.time()
    trainer = make_protocol(HE_FM_FT, federal_info, sec_param, algo_param=None)

    # 联邦计算结果
    fed_grads = trainer.exchange(guest_theta, guest_v, guest_features, guest_labels)
    end = time.time()
    print(f'time is {end - start}')


if __name__ == '__main__':
    test_fm_ft()
