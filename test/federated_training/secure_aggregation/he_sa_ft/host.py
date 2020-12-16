
import numpy as np
from flex.constants import HE_SA_FT
from flex.api import make_protocol

from test.fed_config_example import fed_conf_host


def test():
    theta = [
        [np.random.uniform(-1, 1, (2, 4)).astype(np.float32), np.random.uniform(-1, 1, (2, 6)).astype(np.float32)],
        [np.random.uniform(-1, 1, (2, 8)).astype(np.float32)]]
    print(theta)

    federal_info = fed_conf_host

    sec_param = {
        "he_algo": "paillier",
        "he_key_length": 1024,
        "key_exchange_size": 2048
    }

    trainer = make_protocol(HE_SA_FT, federal_info, sec_param)
    result = trainer.exchange(theta)
    print(result)


if __name__ == '__main__':
    test()
