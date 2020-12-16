import os

import numpy as np
import pandas as pd

from flex.api import make_protocol
from flex.constants import IV_FFS

from test.fed_config_example import fed_conf_host


def test():
    federal_info = fed_conf_host

    sec_param = {
        "he_algo": 'paillier',
        "he_key_length": 1024
    }

    algo_param = {
        'adjust_value': 0.5
    }

    iv_ffs = make_protocol(IV_FFS, federal_info, sec_param, algo_param)
    table = pd.read_csv(os.path.join(os.path.dirname(__file__), 'shap_finance_c.csv'), nrows=300)
    data = pd.Series(table['Occupation'])
    split_info = {'split_points': np.array([0.0, 1.5, 3.01, 4.15, 6.02, 7.04, 8.28, 10.1])}
    iv_value = iv_ffs.exchange(feature=data, is_continuous=True, split_info=split_info)
    print(iv_value)


if __name__ == '__main__':
    test()
