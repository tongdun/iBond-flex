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
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: main.py
#
#  Create date: 2021/01/08
#
from logging import getLogger
from typing import Dict, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame
from caffeine.feature.feature_engineering.constant_params import feature_params, USE_MULTIPROCESS
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame


class CompareKey(str, Enum):
    iv = "iv"
    ks = "ks"
    relief_score = "relief_score"

class SelectionBase(object):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict):
        """
        Initiate base selection for all methods of hetero selection. 
        Mainly for data type checking, parameters initialization and etc.

        Args:
            meta_param, a dict of meta parameters.
        """
        self.logger = getLogger(self.__class__.__name__)
        self._meta_params = meta_params
        self._d_data = meta_params.get('d_data')
        self.common_params = meta_params.get('train_param').get('common_params', {})
        self.down_feature_num = self.common_params.get('down_feature_num', feature_params['down_feature_num'])
        self.max_num_col = self.common_params.get('max_num_col', feature_params['max_num_col'])
        self.max_feature_num_col = self.common_params.get('max_feature_num_col', feature_params['max_feature_num_col'])
        self.logger.info(f'*** _d_data {self._d_data}')
        self.use_multiprocess = self.common_params.get('use_multiprocess', USE_MULTIPROCESS)

    def select_feature(self, data: Optional[IBondDataFrame]=None, \
                            data_attributes: Optional[FeatureDataFrame]=None) \
                            -> Optional[Tuple[IBondDataFrame, FeatureDataFrame]]:
        """
        Select feature base.

        Args:
            data: IBondDataFrame, optional.
            data_attributes: FeatureDataFrame, optional.

        Return:
            IBondDataFrame and FeatureDataFrame.
        """
        raise NotImplementedError()

    def _local_select(self, *args):
        """
        Local selection base.

        Args:
            args: list, depends on algorithm.
        """
        raise NotImplementedError()

    def _federation_select(self, *args):
        """
        Federation selection base for all participants.

        Args:
            args: list, depends on algorithm.
        """
        raise NotImplementedError()

    def _guest2host_select(self, *args):
        """
        Guest aids host for hetero selection, possible by label.

        Args:
            None.
        """
