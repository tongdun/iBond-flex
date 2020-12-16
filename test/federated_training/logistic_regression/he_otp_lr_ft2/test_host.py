import numpy as np
from numpy.random import RandomState

from flex.api import make_protocol
from flex.constants import HE_OTP_LR_FT2
from test.fed_config_example import fed_conf_host
from test.utils import almost_equal


def test_he_otp_lr_ft2():
    federal_info = fed_conf_host

    sec_param = {
        "he_algo": 'paillier',
        "he_key_length": 1024
    }

    prng = RandomState(0)
    guest_theta = prng.uniform(-1, 1, (6,))
    guest_features = prng.uniform(-1, 1, (32, 6))
    guest_labels = prng.randint(0, 2, (32,))

    host_theta = prng.uniform(-1, 1, (6,))
    host_features = prng.uniform(-1, 1, (32, 6))

    def calu_grad(host_theta, host_features, guest_theta, guest_features, guest_labels):
        u2 = host_theta.dot(host_features.T)
        u1 = guest_theta.dot(guest_features.T)
        u = u1 + u2
        h_x = 1 / (1 + np.exp(-u))
        diff_y = guest_labels - h_x

        batch_size = host_features.shape[0]
        grads = (-1 / batch_size) * (diff_y.dot(host_features))

        return grads

    trainer = make_protocol(HE_OTP_LR_FT2, federal_info, sec_param, algo_param=None)

    # 联邦计算结果
    fed_grads = trainer.exchange(host_theta, host_features)

    # 本地计算结果
    local_grads = calu_grad(host_theta, host_features, guest_theta, guest_features, guest_labels)

    assert almost_equal(fed_grads, local_grads)




