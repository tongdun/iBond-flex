import numpy as np
from numpy.random import RandomState

from flex.api import make_protocol
from flex.constants import HE_OTP_LR_FT1
from test.fed_config_example import fed_conf_guest
from test.utils import almost_equal


def test_he_otp_lr_ft1():
    federal_info = fed_conf_guest

    sec_param = [['paillier', {"key_length": 1024}], ]

    prng = RandomState(0)
    guest_theta = prng.uniform(-1, 1, (6,))
    guest_features = prng.uniform(-1, 1, (32, 6))
    guest_labels = prng.randint(0, 2, (32,))

    host_theta = prng.uniform(-1, 1, (6,))
    host_features = prng.uniform(-1, 1, (32, 6))

    def calu_grad(guest_theta, guest_features, guest_labels, host_theta, host_features):
        u2 = host_theta.dot(host_features.T)
        u1 = guest_theta.dot(guest_features.T)
        u = u1 + u2
        h_x = 1 / (1 + np.exp(-u))

        batch_size = guest_features.shape[0]
        grads = (-1 / batch_size) * ((guest_labels - h_x).dot(guest_features))
        return h_x, grads

    # print(guest_theta, guest_features, guest_labels)

    trainer = make_protocol(HE_OTP_LR_FT1, federal_info, sec_param, algo_param=None)

    # 联邦计算结果
    fed_h_x, fed_grads = trainer.exchange(guest_theta, guest_features, guest_labels)

    # 本地计算结果
    local_h_x, local_grads = calu_grad(guest_theta, guest_features, guest_labels, host_theta, host_features)

    assert almost_equal(fed_h_x, local_h_x)
    assert almost_equal(fed_grads, local_grads)



