#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name: common                                                                          
#                                                                                              
#  Create date: 2020/11/24                                                                               
#
import numpy as np
from pydantic import BaseModel, Field, PositiveInt, confloat
from typing import Dict, List, Optional, Tuple, Union

from caffeine.feature.feature_engineering.constant_params import feature_params
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.selection.base_select import SelectionBase
from caffeine.feature.mixins import FLEXUser, IonicUser
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame


class IvConfig(BaseModel):
    iv_thres: confloat(gt=0, le=10) = feature_params['iv_thres']
    top_k: Union[None, PositiveInt] = None

    class Config:
        schema_extra = {
            'expose': ["iv_thres"]
        }

class HeteroIvSelectionBase(SelectionBase, FLEXUser, IonicUser):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Optional[Dict]=dict()):
        """
        Common init ionic channels for sync progress from guest, and 
        exchange messages between guest and host, finally broadcast the 
        ending message to coordinator.

        Args:
            meta_params, including federation, channel params and protocols.
            config: dict, optional, contains algorithm params, probably for flex. 
        """
        super().__init__(meta_params)
        IvConfig.parse_obj(config)
        self.iv_thres = config.get('iv_thres', feature_params['iv_thres'])
        self.top_k = config.get('top_k', feature_params['iv_top_k'])



