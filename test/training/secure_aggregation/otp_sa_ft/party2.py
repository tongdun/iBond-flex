
import numpy as np
from flex.api import make_protocol
from flex.constants import OTP_SA_FT

from test.fed_config_example import fed_conf_guest2


def test():
    theta = [
        [np.random.uniform(-1, 1, (2, 4)).astype(np.float32), np.random.uniform(-1, 1, (2, 6)).astype(np.float32)],
        [np.random.uniform(-1, 1, (2, 8)).astype(np.float32)]]
    print(theta)

    federal_info = fed_conf_guest2

    sec_param = [('onetime_pad', {"key_length": 512})]

    trainer = make_protocol(OTP_SA_FT, federal_info, sec_param, None)
    result = trainer.exchange(theta)
    print(result)


if __name__ == '__main__':
    test()
