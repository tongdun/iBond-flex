from typing import Dict, Union, List, Tuple, Optional
from functools import partial

import pandas as pd
import numpy as np

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame

@ClassMethodAutoLog()
def partial_map(func, kwargs: Dict):
    return partial(func, **kwargs)

@ClassMethodAutoLog()
def label_count(label: Union[pd.Series, np.ndarray]) \
                -> Tuple[int, int, float, float]:
    len_ = len(label)
    bad_all_cnt = sum(label)
    good_all_cnt = len_ - bad_all_cnt
    good_ratio = good_all_cnt * 1. / len_
    bad_ratio = bad_all_cnt * 1. / len_
    return good_all_cnt, bad_all_cnt, good_ratio, bad_ratio


@ClassMethodAutoLog()
def make_output(data: Optional[IBondDataFrame]=None, 
                verify_data: Optional[FeatureDataFrame]=None,
                data_attrib: Optional[Dict]=dict()
        ) -> Dict:
    output = dict()
    output['data'] = data
    output['verify_data'] = verify_data
    if data is None:
        output['data_desc'] = None
    else:
        output['data_desc'] = data.data_desc
    output['data_attrib'] = data_attrib
    return output