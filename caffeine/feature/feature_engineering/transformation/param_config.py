import os
from enum import Enum
from flex.utils.check_security_params import (
    valid_otp_pn_fl,
    valid_he_dt_fb,
    valid_iv_ffs)
from pydantic import BaseModel, conint, Field, validator
from typing import List, Dict, Union, Optional, Any

from caffeine.feature.feature_engineering.transformation.binning.param_config import (BinConfig,
                                                                                      HeteroBinMethod, LocalBinMethod,
                                                                                      BinMethod, EquifrequentBinConfig)
from caffeine.feature.feature_engineering.transformation.mapping.param_config import (IntegerMapConfig,
                                                                                      WOEConfig, OneHotMapConfig,
                                                                                      HeteroMapMethod, LocalMapMethod)

try:
    from safe.dags.dag_infos import DAG_PARAMS_INFO
except:
    DAG_PARAMS_INFO = {}

class LocalProcessMethod(str, Enum):
    local = 'local'


class HeteroProcessMethod(str, Enum):
    hetero = 'hetero'


class CommonParams(BaseModel):
    use_multiprocess: bool = False


class HeteoConfigs(BaseModel):
    '''
    The namespace of this function in dag infos is "HeteroTransformConfigs".
    '''
    dt_bin: Union[None, BinConfig] = BinConfig()
    woe_map: Union[None, WOEConfig] = WOEConfig()

    class Config:
        default_exposion = ["dt_bin", "woe_map"]
        schema_extra = {
            'expose': DAG_PARAMS_INFO['HeteroTransformConfigs'].get(os.getenv('DAG_NAME'), default_exposion) \
                if (os.getenv('DAG_NAME', None) and DAG_PARAMS_INFO.get('HeteroTransformConfigs', None)) \
                else default_exposion
        }


class LocalConfigs(BaseModel):
    '''
    The namespace of this function in dag infos is "LocalTransformConfigs".
    '''
    equifrequent_bin: Union[None, EquifrequentBinConfig] = EquifrequentBinConfig()
    equidist_bin: Union[None, BinConfig] = BinConfig()
    integer_map: Union[None, IntegerMapConfig] = IntegerMapConfig()
    onehot_map: Union[None, OneHotMapConfig] = OneHotMapConfig()

    class Config:
        default_exposion = ["equifrequent_bin", "integer_map"]
        schema_extra = {
            'expose': DAG_PARAMS_INFO['LocalTransformConfigs'].get(os.getenv('DAG_NAME'), default_exposion) \
                if (os.getenv('DAG_NAME', None) and DAG_PARAMS_INFO.get('LocalTransformConfigs', None)) \
                else default_exposion
        }


class Configs(HeteoConfigs, LocalConfigs):
    pass

class HeteroSecurity(BaseModel):
    '''
    The security info for SampleAlignment Operation.

    encryption0: [None or class Encryption0]. Default is Encryption0().
            If Encryption0, Check the class Encryption0.
    '''
    OTP_PN_FL: List = [["onetime_pad", {"key_length": 512}],]
    HE_DT_FB: List = [["paillier", {"key_length": 1024}],]
    IV_FFS: List = [["paillier", {"key_length": 1024}],]

    _otp_pn_fl = validator('OTP_PN_FL', allow_reuse=True)(valid_otp_pn_fl)
    _he_dt_fb = validator('HE_DT_FB', allow_reuse=True)(valid_he_dt_fb)
    _iv_ffs = validator('IV_FFS', allow_reuse=True)(valid_iv_ffs)

class FeatureTransformTrainParams(BaseModel):
    process_method: Union[LocalProcessMethod, HeteroProcessMethod] = 'hetero'
    pipeline: Union[List[Union[HeteroBinMethod, HeteroMapMethod]], 
                    List[Union[LocalBinMethod, LocalMapMethod]]] = ['dt_bin', 'woe_map']
    configs: Configs = Configs()
    common_params: CommonParams = CommonParams()

class FeatureTransformPredictParams(BaseModel):
    configs: Any = None

class FeatureTransformMetaParams(BaseModel):
    train_param: Optional[FeatureTransformTrainParams]
    predict_param: Optional[FeatureTransformPredictParams]
    security_param: Optional[HeteroSecurity]
    federal_info: Optional[Dict]

class HeteroFeatureTransformTrainParams(BaseModel):
    process_method: HeteroProcessMethod = 'hetero'
    pipeline: List[Union[HeteroBinMethod, HeteroMapMethod]] = ['dt_bin', 'woe_map']
    configs: HeteoConfigs = HeteoConfigs()
    common_params: CommonParams = CommonParams()

    class Config:
        schema_extra = {
            'expose': ["configs"]
        }

class HeteroFeatureTransformMetaParams(BaseModel):
    train_param: Optional[HeteroFeatureTransformTrainParams]
    predict_param: Optional[FeatureTransformPredictParams]
    security_param: Optional[HeteroSecurity]
    federal_info: Optional[Dict]

class LocalMethod(str, Enum):
    equifrequent_bin = 'equifrequent_bin'
    equidist_bin = 'equidist_bin'
    integer_map = 'integer_map'
    onehot_map = 'onehot_map'
    
class LocalFeatureTransformTrainParams(BaseModel):
    process_method: LocalProcessMethod = 'local'
    pipeline: List[LocalMethod] = ['equifrequent_bin', 'integer_map', 'onehot_map']
    configs: LocalConfigs = LocalConfigs()
    common_params: CommonParams = CommonParams()

    class Config:
        schema_extra = {
            'expose': ["configs"]
        }

class LocalFeatureTransformMetaParams(BaseModel):
    train_param: Optional[LocalFeatureTransformTrainParams]
    predict_param: Optional[FeatureTransformPredictParams]
    security_param: Optional[Any] = None
    federal_info: Optional[Dict]