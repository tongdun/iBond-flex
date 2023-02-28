import os

import pandas as pd

from flex.api import make_protocol
from flex.constants import IV_FFS

from flex_test.fed_config_example import fed_conf_guest


def test():
    federal_info = fed_conf_guest

    sec_param = [['paillier', {"key_length": 1024}]]

    algo_param = {
        "adjust_value": 1.0,
        "iv_thres": 0.02
    }

    iv_ffs = make_protocol(IV_FFS, federal_info, sec_param, algo_param)
    table = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../../test_data/shap_finance_c.csv'), nrows=300)
    label = pd.Series(table['Label'])
    iv_ffs.exchange(label=label)


if __name__ == '__main__':
    test()
