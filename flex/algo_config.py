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
"""
This file set same default parameters of protocols
"""

# ======================preprocessing================
# DT, bin split
HE_DT_FB = {
    'category_threshold': 5
}

# CHI_BIN
HE_CHI_FB = {
    "max_bin_num": 6,
    "min_bin_num": 4,
    "frequent_value": 50
}

# KS_FFS
KS_FFS_PARAM = {
    "ks_thres": 0.01
}

# IV_FFS
IV_FFS_PARAM = {
    "iv_thres": 0.02,
    "adjust_value": 1.0
}


# ======================training=====================
# HE_GB_FT
HE_GB_FT_PARAM = {
    'min_samples_leaf': 1,
    'lambda_': 0
}

# HE_FM_FT
HE_FM_FT_PARAM = {
    'clip_param': 2.0
}
