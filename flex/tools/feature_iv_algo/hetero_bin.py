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
from flex.tools.feature_iv_algo.base_bin import Bin
import math


class HeteroBin(Bin):
    def __init__(self):  # , bin_param
        super(HeteroBin, self).__init__()  # bin_param

    # host calc good/bad num value when y is decoded
    def en_good_bad_calc(self, y):
        good_num = []
        bad_num = []
        for i in self.bin_result['index']:
            good_num.append(sum(y[i]))
            bad_num.append(len(i) - sum(y[i]))
        # list to numpy
        good_num = np.array(good_num)
        bad_num = np.array(bad_num)
        return good_num, bad_num

    def calc_woe_iv_encode(self, y, good_num, bad_num, decryptor, adjust_value):
        # num of positive/negative samples
        good_all_count, bad_all_count = self.label_count(y)
        woe_value = []
        iv = 0
        for i, good_num_value in enumerate(good_num):
            # calc woe
            good_num_value = decryptor.decrypt(good_num_value)
            bad_num_value = decryptor.decrypt(bad_num[i])
            if good_num_value == 0 or bad_num_value == 0:
                calc_value = math.log((bad_num_value / bad_all_count + adjust_value) /
                                      (good_num_value / good_all_count + adjust_value))
            else:
                calc_value = math.log((bad_num_value / bad_all_count) /
                                      (good_num_value / good_all_count))
            woe_value.append(calc_value)
            # calc iv
            iv += ((bad_num_value / bad_all_count) -
                   (good_num_value / good_all_count)) * calc_value
        return woe_value, iv
