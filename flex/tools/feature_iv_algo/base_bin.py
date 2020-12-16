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

import numpy as np


class Bin(object):
    """
    in this method we change origin data to bins
    all bin's message return to dict
    """

    def __init__(self):
        self.bin_result = dict()
        self.bin_hist = dict()

    def get_bin_hist(self, data, split_info):
        self.bin_hist['split_points'] = split_info.get('split_points')
        self.trans_bin_dict(data)
        self.bin_result['index'] = self.bin_hist['index']
        self.bin_result['data'] = self.bin_hist['data']
        self.bin_result['split_points'] = self.bin_hist['split_points']

    # calc discrete feature bin message
    def get_discrete_bin(self, data):
        data = data.values
        index_list, data_list = [], []
        threshold = list(np.unique(data))
        for i in threshold:
            index_list.append(np.where(data == i)[0])
            data_list.append(np.array([i]))
        self.bin_result['index'] = index_list
        self.bin_result['data'] = data_list
        self.bin_result['splint_points'] = threshold

    # get good/bad nums of each bin
    def good_bad_calc(self, y):
        """
        y: input label mess, y must be array
        """
        good_num = []
        bad_num = []
        for i in self.bin_result['index']:
            good_num.append(sum(y[i]))
            bad_num.append(len(i) - sum(y[i]))
        self.bin_result['good_num'] = good_num
        self.bin_result['bad_num'] = bad_num

    def label_count(self, y):
        """
        calc good、bad nums of label, y(label)
        """
        good_all_count = sum(y)
        bad_all_count = len(y) - sum(y)
        return good_all_count, bad_all_count

    def trans_bin_dict(self, data):
        """
        Args:
            data: input data
        Returns: update bin hist mess
        """
        self.bin_hist['index'] = []
        self.bin_hist['data'] = []
        edges = self.bin_hist['split_points']
        for i, value in enumerate(edges):
            if i != 0:
                value_l = self.bin_hist['split_points'][i - 1]
                value_r = self.bin_hist['split_points'][i]
                func = lambda x: value_l < x <= value_r
            else:
                # data value less than min value
                value_j = self.bin_hist['split_points'][0]
                func = lambda x: x <= value_j
            index_f = data[data.apply(func)]
            self.bin_hist['data'].append(index_f.values.reshape(1, -1)[0])
            self.bin_hist['index'].append(np.array(index_f.index))
        # data value more than max value
        value_e = self.bin_hist['split_points'][-1]
        func = lambda x: x > value_e
        index_f = data[data.apply(func)]
        self.bin_hist['data'].append(index_f.values.reshape(1, -1)[0])
        self.bin_hist['index'].append(np.array(index_f.index))

