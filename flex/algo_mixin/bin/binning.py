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


class EquiFrequentBin(object):
    """
    Apply equal frequent to bin split
    """
    def __init__(self, bin_num: int):
        """
        Bin split parameters inits

        Args:
            bin_num: num of split
        """
        self.bin_num = bin_num

    def get_split_index(self, data: np.ndarray) -> np.ndarray:
        """
        This method mainly use equal frequent to bin split

        Args:
            data: bin split datasets

        Return:
             np.array, split point
        """
        # percentiles of each feature
        num_bin = min(self.bin_num, len(np.unique(data)))
        percentile = np.linspace(0, 1, num_bin + 1)[1:-1]*100

        # bin threshold of feature
        describe = np.percentile(data[~np.isnan(data)], percentile)
        describe = np.sort(np.unique(np.round(describe, 4)))
        return describe
