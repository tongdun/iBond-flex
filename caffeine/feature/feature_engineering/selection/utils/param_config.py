from enum import Enum
from typing import Union, List, Optional, Dict, Any

from pydantic import BaseModel, Field

from .hetero_param_config import (HeteroProcessMethod, HeteroSelectionMethod, 
        HeteroSecurity,
        HeteroConfigs, CommonParams)

class Security(HeteroSecurity):
    pass
class FeatureSelectionTrainParams(BaseModel):
    process_method: Union[HeteroProcessMethod] = Field('hetero')
    pipeline: List[Union[HeteroSelectionMethod]]
    configs: HeteroConfigs = HeteroConfigs()
    common_params: CommonParams = CommonParams()

class FeatureSelectionPredictParams(BaseModel):
    configs: Any = None

class FeatureSelectionMetaParams(BaseModel):
    train_param: Optional[FeatureSelectionTrainParams]
    predict_param: Optional[FeatureSelectionPredictParams]
    security_param: Optional[Security]
    federal_info: Optional[Dict]