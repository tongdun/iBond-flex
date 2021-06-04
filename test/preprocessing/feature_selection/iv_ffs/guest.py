import os

import random

import numpy as np
import pandas as pd

from flex.api import make_protocol
from flex.constants import IV_FFS

federation = ["zhibang-d-011040", "zhibang-d-011041", "zhibang-d-011042", "zhibang-d-014031"]


def test():
    federal_info = {
        "server": "localhost:6001",
        "session": {
            "role": "guest",
            "local_id": federation[1],
            "job_id": 'test_job'},
        "federation": {
            "host": [federation[0]],
            "guest": [federation[1]],
            "coordinator": [federation[2]]}
    }

    sec_param = [['paillier', {"key_length": 1024}]]

    algo_param = {
        "adjust_value": 1.0,
        "iv_thres": 0.02
    }

    label = np.array(list(1 if random.random() >= 0.5 else 0 for i in range(10000)))
    # table = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../../test_data/hive_test.csv'))  # , nrows=300
    # label = np.array(table['test_sample_process_op_0_out_guest_20210104_01.y'])
    iv_ffs = make_protocol(IV_FFS, federal_info, sec_param, algo_param)
    iv_ffs.pre_exchange(label=label)
    iv_ffs.exchange(label=label)


if __name__ == '__main__':
    test()
