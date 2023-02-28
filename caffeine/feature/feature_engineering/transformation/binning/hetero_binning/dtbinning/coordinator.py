from typing import Dict, Optional, Tuple

from flex.constants import OTP_PN_FL

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.constant_params import feature_params
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from .common import HeteroDtBinBase


class HeteroDtBinCoord(HeteroDtBinBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Dict, context: Context, model_info: Optional[Dict]=None):
        """
        Common init ionic channels for sync progress from guest, and 
        exchange messages between guest and host, finally broadcast the 
        ending message to coordinator.

        Args:
            meta_params, including federal_info, channel params and protocols.
            config: dict, optional, contains algorithm params, probably for flex. 
        """
        super().__init__(meta_params, config, context, model_info)

    @ClassMethodAutoLog()
    def fit(self, data: IBondDataFrame, data_attributes: FeatureDataFrame) -> \
            Tuple[IBondDataFrame, FeatureDataFrame]:         
        """
        Binning method for training data at coordinator.

        Args:
            None.       
        """   
        self._protocols[OTP_PN_FL].param_negotiate('max')
        data_attributes['iv'] = []
        return data, data_attributes