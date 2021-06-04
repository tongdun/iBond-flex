import os

import numpy as np
import pandas as pd

from flex.api import make_protocol
from flex.constants import IV_FFS

federation = ["zhibang-d-011040", "zhibang-d-011041", "zhibang-d-011042", "zhibang-d-014031"]


def test():
    federal_info = {
        "server": "localhost:6001",
        "session": {
            "role": "host",
            "local_id": federation[0],
            "job_id": 'test_job'},
        "federation": {
            "host": [federation[0], federation[2]],
            "guest": [federation[1]],
            "coordinator": []}
    }

    sec_param = [['paillier', {"key_length": 1024}]]

    algo_param = {
        "adjust_value": 1.0,
        "iv_thres": 0.02
    }

    iv_ffs = make_protocol(IV_FFS, federal_info, sec_param, algo_param)
    table = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../../test_data/hive_test.csv'))  # , nrows=300
    feature = pd.Series(table['test_sample_process_op_0_out_host_20210104_01.occupation'])  # Race  Occupation Age
    split_info = {'split_points': np.array([-0.61320213, -0.37735887,  0.09432765,  0.80185742,  1.03770068])}
    en_labels = iv_ffs.pre_exchange()
    iv_value = iv_ffs.exchange(feature=feature, en_labels=en_labels, is_category=False, data_null=False, split_info=split_info)
    print(iv_value)


if __name__ == '__main__':
    test()
