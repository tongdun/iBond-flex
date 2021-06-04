

import os
import math

import numpy as np
import pandas as pd

from flex.api import make_protocol
from flex.constants import IV_FFS

from test.fed_config_example import fed_conf_host


def test_iv_ffs():
    federal_info = fed_conf_host

    sec_param = [['paillier', {"key_length": 1024}]]

    algo_param = {
        "adjust_value": 1.0,
        "iv_thres": 0.02
    }

    def iv_calu(label, data, split_info):
        bin_hist = dict()
        bin_hist['index'] = []
        edges = split_info['split_points']
        for i, value in enumerate(edges):
            if i != 0:
                value_l = split_info['split_points'][i - 1]
                value_r = split_info['split_points'][i]
                func = lambda x: value_l < x <= value_r
            else:
                value_j = split_info['split_points'][0]
                func = lambda x: x <= value_j
            index_f = data[data.apply(func)]
            bin_hist['index'].append(np.array(index_f.index))
        value_e = split_info['split_points'][-1]
        func = lambda x: x > value_e
        index_f = data[data.apply(func)]
        bin_hist['index'].append(np.array(index_f.index))

        good_num = []
        bad_num = []
        for i in bin_hist['index']:
            good_num.append(sum(label[i]))
            bad_num.append(len(i) - sum(label[i]))
        good_num = np.array(good_num)
        bad_num = np.array(bad_num)

        good_all_count = sum(label)
        bad_all_count = len(label) - sum(label)
        iv = 0
        for i, good_num_value in enumerate(good_num):
            if good_num_value == 0 or bad_num[i] == 0:
                calc_value = math.log((bad_num[i] / bad_all_count + algo_param['adjust_value']) /
                                      (good_num_value / good_all_count + algo_param['adjust_value']))
            else:
                calc_value = math.log((bad_num[i] / bad_all_count) /
                                      (good_num_value / good_all_count))
            iv += ((bad_num[i] / bad_all_count) -
                   (good_num_value / good_all_count)) * calc_value
        return iv

    iv_ffs = make_protocol(IV_FFS, federal_info, sec_param, algo_param)
    table = pd.read_csv(os.path.join(os.path.dirname(__file__), 'shap_finance_c.csv'), nrows=300)
    data = pd.Series(table['Occupation'])
    label = pd.Series(table['Label'])
    split_info = {'split_points': np.array([0.0, 1.5, 3.01, 4.15, 6.02, 7.04, 8.28, 10.1])}
    # 联邦计算的iv值
    iv_value = iv_ffs.exchange(feature=data, is_continuous=True, split_info=split_info)
    # 本地计算的iv值
    local_iv_value = iv_calu(label, data, split_info)

    assert iv_value == local_iv_value


