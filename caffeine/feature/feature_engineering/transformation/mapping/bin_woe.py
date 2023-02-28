from typing import Dict, Union, List, Tuple

import pandas as pd
import numpy as np
import math

from caffeine.utils import ClassMethodAutoLog
from caffeine.feature.feature_engineering.utils import partial_map
from caffeine.feature.feature_engineering.constant_params import EPS


class BinWoe(object):
    # @ClassMethodAutoLog()
    def compute_woe_iv(self, good_num: List[float], bad_num: List[float], good_all_cnt: int, \
                 bad_all_cnt: int) -> Tuple[List[float], float]:
        """
        Compute woe and iv.

        Args:
            good_num: list, good count samples of each bin.
            bad_num: list, bad count samples of each bin.
            good_all_cnt: int, good samples of the whole dataset.
            bad_all_cnt: int, bad samples of the whole dataset.

        Return:
            woes: list, woe of each bin.
            iv: float, iv value.
        """
        woes = []
        iv = 0.0
        for i, good_i in enumerate(good_num):
            bad_i = bad_num[i]
            if min(good_i, bad_i) < EPS:
                woe = math.log((bad_i / bad_all_cnt + self.adjust_value_woe) /
                                (good_i / good_all_cnt + self.adjust_value_woe))
            else:
                woe = math.log((bad_i / bad_all_cnt) /
                                (good_i / good_all_cnt))
                iv += (bad_i / bad_all_cnt - good_i / good_all_cnt) * woe
            woes.append(woe)
        return woes, float(iv)

    # @ClassMethodAutoLog()
    def woe_mono(self, woe: Union[List, np.ndarray], is_category: bool, is_fillna: bool) -> bool:
        """
        Check if woes are monotone or U-shape.

        Args:
            woe: list or np.ndarray
            is_fillna: bool, true if contains np.nan otherwise false

        Return:
            bool: monotone or not.
        """
        self.logger.info(f'>>>>>> check_monotone {self.check_monotone}')
        if self.check_monotone is False:
            return True
        if is_category is True:
            return True
        if is_fillna is False:
            return check_monotone(woe[:-1])
        else:
            return check_monotone(woe)

    # @ClassMethodAutoLog()
    def trans_to_woes(self, data_col: np.ndarray, woes: List, 
                            split_points: List) -> np.ndarray:
        """
        Map data_col to woes.

        Args: 
            data_col: pd.Series or np.ndarray, input column data.
            woes: list, woes corresponding to bins.
            split_points: list, bin edges.

        Return:
            np.ndarray, mapped woe values.
        """
        data_col = data_col.astype(np.float64)
        woes = np.array(woes)
        data = data_col[~np.isnan(data_col)]
        idx = np.digitize(data, split_points, right=True)
        data_col[~np.isnan(data_col)] = woes[idx]
        data_col[np.isnan(data_col)] = woes[-1]
        return data_col


# @ClassMethodAutoLog()
def check_monotone(v: Union[List, np.ndarray]) -> bool:
    """
    Check if array v is monotone or U-shape.

    Args:
        v: np.ndarray or list.

    Return:
        bool.
    """
    result = is_woe_monoton(v)
    if result is False:
        result = is_woe_u_shape(v)
    return result

# @ClassMethodAutoLog()
def is_woe_monoton(v: Union[List, np.ndarray]) -> bool:
    """
    Check the woe is monotonic or not.

    Args:
        v: np.ndarray or list.

    Return:
        bool.
    """
    if len(v) <= 1:
        return True

    ascending = all(x < y for x, y in zip(v, v[1:]))
    descending = all(x > y for x, y in zip(v, v[1:]))
    return ascending or descending

# @ClassMethodAutoLog()
def is_woe_u_shape(v: Union[List, np.ndarray]) -> bool:
    """
    Check if woe is u-shape.

    Args:
        v: np.ndarray or list.

    Return:
        bool.
    """
    if len(v) <= 1:
        return True

    min_idx = np.argmin(v)
    max_idx = np.argmax(v)
    is_concave = False
    is_convex = False
    if min_idx not in [0, len(v)-1]:
        is_convex = all(x > y for x, y in zip(v[:(min_idx+1)], v[1:(min_idx+1)])) and \
                        all(x < y for x, y in zip(v[min_idx:], v[min_idx+1:]))
    if max_idx not in [0, len(v)-1]:
        is_concave = all(x < y for x, y in zip(v[:(max_idx+1)], v[1:(max_idx+1)])) and \
                    all(x > y for x, y in zip(v[max_idx:], v[(max_idx+1):]))       
    
    return is_concave or is_convex