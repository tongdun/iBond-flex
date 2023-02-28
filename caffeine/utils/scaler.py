#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2021 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond
#                                                                                              
#  File name: scaler.py
#                                                                                              
#  Create date:  2021/7/19                                                                    

from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer

from caffeine.utils import IBondDataFrame

scaler_type_lit = ['standard', 'normalize', 'minmax', 'maxabs']


def form_scaler(scaler_type: str, data: Union[np.ndarray, pd.DataFrame, IBondDataFrame]):
    '''

    Args:
        scaler_type: str. Scale method for processing data.
        feats: Union[np.ndarray, pd.DataFrame]. Feature data for training process.

    Returns:
        scaler: Scaler model which store with info of scale.
        processed_data: np.ndarray The trainsformed feature data.

    Examples:
        data
                  ID  Label   Age  ...  CapitalLoss  Hoursperweek  Country
    0          1      0  39.0  ...          0.0          40.0       39
    1          2      0  50.0  ...          0.0          13.0       39

        feats = data.iloc[:, 2:]
        feats
                Age  Workclass  Education-Num  ...  CapitalLoss  Hoursperweek  Country
    0      39.0          7           13.0  ...          0.0          40.0       39
    1      50.0          6           13.0  ...          0.0          13.0       39

    scaler_type = 'standard'
    scaler = scaler_data(scaler_type, feats)
    '''
    if scaler_type not in scaler_type_lit:
        raise Exception(f'Unsupported scaler type {scaler_type}')
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'normalize':
        scaler = Normalizer()
    else:
        raise Exception(f'Unsupported scaler type {scaler_type}')

    if isinstance(data, IBondDataFrame):
        nonfeat_cols = data.data_desc['id_desc'] + data.data_desc['y_desc']
        feat_cols = [n for n in data.columns if n not in nonfeat_cols]
        tmp = data.to_pandas()
        data = tmp[feat_cols]
    scaler.fit(data)
    return scaler  # 这里的特征名称暂时不考虑


def process_data_with_scaler(scaler, data: Union[np.ndarray, pd.DataFrame, IBondDataFrame]) -> Union[
    np.ndarray, pd.DataFrame, IBondDataFrame]:
    '''

    Args:
        scaler: The saved scaler model which store with info of scale.
        data: The data which has the same features like the scaler.

    Returns:
        processed_data: np.ndarray The trainsformed feature data.

    Examples:
        processed_data
        array([[ 0.02717727,  2.14993921,  1.13904693, ..., -0.21567921,
            -0.03328008,  0.29009487],
           [ 0.83104303,  1.46320727,  1.13904693, ..., -0.21567921,
            -2.22708886,  0.29009487]])
    '''
    if isinstance(data, IBondDataFrame):
        nonfeat_cols = data.data_desc['id_desc'] + data.data_desc['y_desc']
        feat_cols = [n for n in data.columns if n not in nonfeat_cols]
        tmp = data.to_pandas()
        processed_data = scaler.transform(tmp[feat_cols])
        tmp[feat_cols] = processed_data
        data._pdf = tmp
    elif isinstance(data, np.ndarray):
        data = scaler.transform(data)
    elif isinstance(data, pd.DataFrame):
        processed_data = scaler.transform(data)
        data.iloc[:, :] = processed_data
    else:
        raise Exception(f'Unsupported data type')
    return data
