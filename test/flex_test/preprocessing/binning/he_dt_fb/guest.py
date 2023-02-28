import os

import pandas as pd
import random
import time
import numpy as np

from flex.api import make_protocol
from flex.constants import HE_DT_FB


from flex_test.fed_config_example import fed_conf_guest


def test():
    # param inits
    federal_info = fed_conf_guest
    sec_param = [['paillier', {"key_length": 1024}]]
    algo_param = {
        "host_party_id": "zhibang-d-011040",
        "max_bin_num": 6,
        "frequent_value": 50
    }

    # dataloader
    # table = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../../test_data/hive_test.csv'))  # , nrows=300
    # label = table['test_sample_process_op_0_out_guest_20210104_01.y'].values
    label = np.array(list(1 if random.random() >= 0.5 else 0 for i in range(10000)))

    time1 = time.time()

    # inits object
    trainer = make_protocol(HE_DT_FB, federal_info, sec_param, algo_param)

    # pre_exchange send encrypted label
    trainer.pre_exchange(label)

    # exchange print result
    result = trainer.exchange(label)
    print(result)
    print(time.time()-time1)


if __name__ == '__main__':
    test()
