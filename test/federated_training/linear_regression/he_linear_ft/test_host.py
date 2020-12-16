import numpy as np
from numpy.random import RandomState

from flex.api import make_protocol
from flex.constants import HE_LINEAR_FT

from test.fed_config_example import fed_conf_host
from test.utils import almost_equal


def test_he_linear_ft():

    # host和guest的随机初始状态相同
    prng = RandomState(0)
    guest_u = np.array(prng.uniform(-1, 1, (8,)))
    host_u = np.array(prng.uniform(-1, 1, (8,)))

    guest_labels = np.array(prng.randint(0, 2, (8,)))

    federal_info = fed_conf_host

    sec_param = {
        "he_algo": 'paillier',
        "he_key_length": 1024
    }

    trainer = make_protocol(HE_LINEAR_FT, federal_info, sec_param, algo_param=None)

    result = trainer.exchange(host_u)
    assert almost_equal(result, guest_u + host_u - guest_labels)

