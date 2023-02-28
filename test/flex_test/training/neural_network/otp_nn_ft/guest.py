import numpy as np
import torch
from flex.constants import OTP_NN_FT
from flex.api import make_protocol

from flex_test.fed_config_example import fed_conf_guest


def test():
    g = [[torch.Tensor(2, 3).uniform_(-1, 1)], [torch.Tensor(3, 2).uniform_(-1, 1)]]

    y = np.random.randint(0, 2, (10,))

    print(g)
    print(y)

    federal_info = fed_conf_guest

    sec_param = [['onetime_pad', {'key_length': 512}], ]

    trainer = make_protocol(OTP_NN_FT, federal_info, sec_param, algo_param=None)
    result = trainer.exchange(g, y)
    print(result)


if __name__ == '__main__':
    test()
