import torch
from flex.constants import OTP_NN_FT
from flex.api import make_protocol

from test.fed_config_example import fed_conf_host


def test():
    g = [[], []]
    for i in range(10):
        g[0].append([[torch.Tensor(2, 3).uniform_(-1, 1)]])
        g[1].append([[torch.Tensor(2, 3).uniform_(-1, 1)]])
    for j in range(10):
        g[0][j].append([torch.Tensor(3, 2).uniform_(-1, 1)])
        g[1][j].append([torch.Tensor(3, 2).uniform_(-1, 1)])

    print(g)

    federal_info = fed_conf_host

    sec_param = [['onetime_pad', {'key_length': 512}], ]

    trainer = make_protocol(OTP_NN_FT, federal_info, sec_param, algo_param=None)
    result = trainer.exchange(g)
    print(result)


if __name__ == '__main__':
    test()
