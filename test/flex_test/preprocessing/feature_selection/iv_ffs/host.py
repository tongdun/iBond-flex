
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
            "role": "host",
            "local_id": federation[0],
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

    feature = pd.Series(list(round(random.random(), 4) for i in range(10000)))
    # table = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../../test_data/shap_finance_c.csv'))  # , nrows=300
    # feature = pd.Series(table['Occupation'])
    iv_ffs = make_protocol(IV_FFS, federal_info, sec_param, algo_param)
    split_info = {'split_points': np.array([0.17, 0.35,  0.53,  0.70,  0.85])}
    en_labels = iv_ffs.pre_exchange()
    iv_value = iv_ffs.exchange(feature=feature, en_labels=en_labels, is_category=False, data_null=False, split_info=split_info)
    print(iv_value)


if __name__ == '__main__':
    test()
