"""dot_mul: host
"""
import torch

from flex.api import make_protocol
from flex.constants import SS_COMPUTE

from test.fed_config_example import fed_conf_host


def test():
    federal_info = fed_conf_host
    sec_param = [['secret_sharing', {'precision': 3}], ]

    algo_param = {
    }

    torch.manual_seed(seed=1111111)
    _ = torch.randint(100, [5, 2])
    A = torch.randint(100, [5, 2])
    A = torch.rand(5, 2)

    protocol = make_protocol(SS_COMPUTE, federal_info, sec_param, algo_param)
    [a_sh, b_sh] = protocol.share_secrets(A)
    print(protocol.substract_rec(a_sh, b_sh))
    print(protocol.le_rec(a_sh, b_sh))


if __name__ == '__main__':
    test()
