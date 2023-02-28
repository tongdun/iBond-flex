import os
from enum import Enum
from flex.constants import OTP_PN_FL
from flex.utils.check_security_params import valid_otp_pn_fl
from pydantic import BaseModel, Field, PositiveInt, validator
from typing import Union, List, Optional, Dict, Any

from caffeine.feature.config import Relief_FMC, Stepwise_FMC
from caffeine.feature.feature_engineering.constant_params import feature_params
from caffeine.feature.feature_engineering.selection.hetero_selection.hetero_iv.common import IvConfig

try:
    from safe.dags.dag_infos import DAG_PARAMS_INFO
except:
    DAG_PARAMS_INFO = {}


class HeteroProcessMethod(str, Enum):
    hetero = "hetero"


class CommonParams(BaseModel):
    use_multiprocess: bool = False
    down_feature_num: PositiveInt = feature_params['down_feature_num']
    max_num_col: PositiveInt = feature_params['max_num_col']
    max_feature_num_col : PositiveInt =  feature_params['max_feature_num_col']


class HeteroSelectionMethod(str, Enum):
    iv = 'Iv'


class HeteroConfigs(BaseModel):
    '''
    The namespace of this function in dag infos is "HeteroSelectionConfigs".
    '''
    Iv: Union[None, IvConfig] = IvConfig()

    class Config:
        default_exposion = ["Iv"]
        schema_extra = {
            'expose': DAG_PARAMS_INFO['HeteroSelectionConfigs'].get(os.getenv('DAG_NAME'), default_exposion) \
                if (os.getenv('DAG_NAME', None) and DAG_PARAMS_INFO.get('HeteroSelectionConfigs', None)) \
                else default_exposion
        }


class HeteroSecurity(BaseModel):
    '''
    The security info for SampleAlignment Operation.

    encryption0: [None or class Encryption0]. Default is Encryption0().
            If Encryption0, Check the class Encryption0.
    '''
    OTP_PN_FL: List = [["onetime_pad", {"key_length": 512}], ]

    _otp_pn_fl = validator('OTP_PN_FL', allow_reuse=True)(valid_otp_pn_fl)


class HeteroFeatureSelectionTrainParams(BaseModel):
    process_method = 'hetero'
    pipeline: List[HeteroSelectionMethod] = ['Iv']
    configs: HeteroConfigs = HeteroConfigs()
    common_params: CommonParams = CommonParams()

    class Config:
        schema_extra = {
            'expose': ["configs"]
        }


class HeteroFeatureSelectionPredictParams(BaseModel):
    configs: Any = None


class HeteroFeatureSelectionMetaParams(BaseModel):
    train_param: Optional[HeteroFeatureSelectionTrainParams]
    predict_param: Optional[HeteroFeatureSelectionPredictParams]
    security_param: Optional[HeteroSecurity]
    federal_info: Optional[Dict]
