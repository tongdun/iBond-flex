import time
import numpy as np
from numpy.random import RandomState

from flex.api import make_protocol
from flex.constants import HE_FM_FT
from test.fed_config_example import fed_conf_host
from test.utils import almost_equal


def test_fm_ft():
    """
    Args:
    host_theta: weight params of FM model's linear term
    host_v: weight params of FM model's embedding term
    host_features: origin dataset
    """
    federal_info = fed_conf_host

    sec_param = []
    prng = RandomState(0)
    batch_size = 32
    feature_num = 1000
    host_theta = prng.uniform(-1, 1, (feature_num, ))
    host_v = prng.uniform(-1, 1, (feature_num, 10))
    host_features = prng.uniform(-1, 1, (batch_size, feature_num))

    start = time.time()
    trainer = make_protocol(HE_FM_FT, federal_info, sec_param, algo_param=None)

    # 联邦计算结果
    fed_grads = trainer.exchange(host_theta, host_v, host_features)
    end = time.time()
    print(f'time is {end - start}')


if __name__ == '__main__':
    test_fm_ft()
