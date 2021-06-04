import os

import pandas as pd
import random
import numpy as np
import time

from flex.api import make_protocol
from flex.constants import HE_DT_FB

from test.fed_config_example import fed_conf_host


def test():
    # param inits
    federal_info = fed_conf_host
    sec_param = [['paillier', {"key_length": 1024}]]
    algo_param = {
        "host_party_id": "zhibang-d-011040",
        "max_bin_num": 6,
        "frequent_value": 50
    }

    # dataloader
    # table = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../../test_data/hive_test.csv'))  # , nrows=300
    # feature = table['test_sample_process_op_0_out_host_20210104_01.occupation'].values
    feature = np.array(list(round(random.random(), 4) for i in range(10000)))

    time1 = time.time()

    # inits object
    trainer = make_protocol(HE_DT_FB, federal_info, sec_param, algo_param)

    # pre_exchange send encrypted label
    en_label = trainer.pre_exchange()

    # exchange calc result
    result = trainer.exchange(en_label, feature)
    print(result)
    print(time.time() - time1)


if __name__ == '__main__':
    test()
