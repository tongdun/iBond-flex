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

from flex.tools.feature_iv_algo.hetero_bin import HeteroBin


def host_calc_bin_res(en_label, data, split_info, data_is_continuous):
    bin_obj = HeteroBin()
    good_bad_nums = []
    if data_is_continuous:
        bin_obj.get_bin_hist(data, split_info)
    else:
        bin_obj.get_discrete_bin(data)
    good_num, bad_num = bin_obj.en_good_bad_calc(en_label)
    bin_result = bin_obj.bin_result
    good_bad_nums.append(good_num)
    good_bad_nums.append(bad_num)
    return bin_result, good_bad_nums


def guest_iv_woe_value(label, good_bad_nums, decryptor, adjust_value):
    good_nums = good_bad_nums[0]
    bad_nums = good_bad_nums[1]
    bin_obj = HeteroBin()
    woe_value, iv_value = bin_obj.calc_woe_iv_encode(y=label,
                                                     good_num=good_nums,
                                                     bad_num=bad_nums,
                                                     decryptor=decryptor,
                                                     adjust_value=adjust_value)
    return woe_value, iv_value

