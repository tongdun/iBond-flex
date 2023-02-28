#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import math
from typing import Dict, List, Tuple

import numpy as np

from ...cores.encrypt_model import EncryptModel


class Bin(object):
    """
    In this method we change origin data to bins, all bin's message return to dict
    """

    def __init__(self):
        self.bin_result = dict()
        self.bin_hist = dict()

    def get_bin_hist(self,
                     feature: np.ndarray,
                     split_list: List) -> None:
        """
        If feature's type is consequent, get binning histogram results.
        """
        self.bin_hist['split_points'] = split_list
        self.trans_bin_dict(feature)
        self.bin_result['index'] = self.bin_hist['index']
        self.bin_result['data'] = self.bin_hist['data']
        self.bin_result['split_points'] = self.bin_hist['split_points']

    def get_discrete_bin(self, feature: np.ndarray) -> None:
        """
        If feature's type is category, get discrete binning info.
        """
        self.bin_result = dict()
        threshold = sorted(np.unique(feature[~np.isnan(feature)]))
        index_list, data_list = [], []
        for i in threshold:
            idx = np.where(feature == i)[0]
            index_list.append(idx)
            data_list.append(np.array([i]))
        self.bin_result['index'] = index_list
        self.bin_result['data'] = data_list
        self.bin_result['splint_points'] = threshold

    def get_none_type_bin(self, feature: np.ndarray) -> None:
        """
        Func handles the situation which feature exists missing values.
        """
        idx = (np.isnan(feature))
        if np.sum(idx) > 0:
            self.bin_result['index'].append(idx)
            self.bin_result['data'].append(None)

    @staticmethod
    def label_count(label: np.ndarray) -> [int, int]:
        """
            calc good and bad nums of label.
        """
        bad_all_count = sum(label)
        good_all_count = len(label) - sum(label)
        return good_all_count, bad_all_count

    def trans_bin_dict(self, data: np.ndarray) -> None:
        """
        Args:
            data: input data

        Returns: update bin hist mess
        """
        self.bin_hist['index'] = []
        self.bin_hist['data'] = []
        self.bin_hist['split_points'] = np.insert(self.bin_hist['split_points'], 0, -np.inf)
        self.bin_hist['split_points'] = np.append(self.bin_hist['split_points'], np.inf)
        edges = self.bin_hist['split_points']
        for i, value in enumerate(edges):
            if i != 0:
                value_l = self.bin_hist['split_points'][i - 1]
                value_r = self.bin_hist['split_points'][i]
                index_f = np.where(((data <= value_r) & (data > value_l)))[0]
                self.bin_hist['data'].append(data[index_f])
                self.bin_hist['index'].append(index_f)
        self.bin_hist['split_points'] = self.bin_hist['split_points'][1:-1]


class HeteroBin(Bin):
    def __init__(self):
        super(HeteroBin, self).__init__()

    def en_good_bad_calc(self, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Host calc good/bad num value when label is encrypted.
        """
        total_num, good_num, bad_num = [], [], []
        for i in self.bin_result['index']:
            total_num.append(len(i))
            bad_num.append(sum(label[i]))
            good_num.append(len(i) - bad_num[-1])
        # list to numpy
        good_num = np.asarray(good_num)
        bad_num = np.asarray(bad_num)
        total_num = np.asarray(total_num)
        return good_num, bad_num, total_num

    def calc_woe_iv(self,
                    y: np.ndarray,
                    total_num: np.ndarray,
                    en_bad_num: np.ndarray,
                    decrypter: EncryptModel,
                    adjust_value: float) -> [List, List]:
        """
        Guest calculates the iv and woe value.
        """
        # num of positive/negative samples
        good_all_count, bad_all_count = self.label_count(y)
        woe_value, iv = [], []
        bad_num = np.asarray([decrypter.decrypt(bad_num_value) for bad_num_value in en_bad_num])
        good_num = total_num - bad_num
        for i, good_num_value in enumerate(good_num):
            # calc woe value
            if good_num_value == 0 or bad_num[i] == 0:
                calc_value = math.log((bad_num[i] / bad_all_count + adjust_value) /
                                      (good_num_value / good_all_count + adjust_value))
            else:
                calc_value = math.log((bad_num[i] / bad_all_count) /
                                      (good_num_value / good_all_count))
            woe_value.append(calc_value)
            # calc iv value
            iv.append(((bad_num[i] / bad_all_count) -
                       (good_num_value / good_all_count)) * calc_value)
        return woe_value, iv

