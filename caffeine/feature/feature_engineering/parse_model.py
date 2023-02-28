from typing import Dict, Union, List, Tuple

import pandas as pd 

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame
from caffeine.model.base_model import JsonModel
from .ibond_feature_df import FeatureDataFrame


@ClassMethodAutoLog()
def parse_model_attribute(model_info: Dict) -> pd.DataFrame:
    """
    Parse feature-processing model to get data attributes 
    like name, is_category and is_fillna.

    Args:
        model_info: dict, feature-processing saved model.

    Return:
        data_atrrib: pd.DataFrame, with columns [name, is_category, is_fillna].
    """
    model_info = JsonModel.parse_obj(model_info)
    model = model_info.modules[0].params['model']['feature_process']

    name = []
    is_category = []
    is_fillna = []
    for key, val in model.items():
        name.append(key)
        is_category.append(val.get('is_category'))
        is_fillna.append(val.get('is_fillna'))
    data_atrrib = pd.DataFrame({'name': name, 
                                'is_category': is_category,
                                'is_fillna': is_fillna})
    return data_atrrib

@ClassMethodAutoLog()
def trans_feature_df(data_atrrib: pd.DataFrame, data_desc: Dict) -> FeatureDataFrame:
    """
    Transform data_atrrib to FeatureDataFrame object.

    Args:
        data_atrrib: pd.DataFrame, containing feature name, is_category and is_fillna.
        data_desc: dict, containing data description.

    Return:
        data_statistics: FeatureDataFrame.
    """
    data_statistics = FeatureDataFrame(data_atrrib, data_desc)
    return data_statistics

