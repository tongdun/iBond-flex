from typing import Dict, List, Optional, Tuple
from logging import getLogger

import pandas as pd

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.check_d_type import CheckDTypeCoord
from caffeine.feature.feature_engineering.utils import make_output
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from .common import FeatureSelectionBase
from .hetero_selection.hetero_iv.coordinator import HeteroIvSelectionCoord


METHOD_NAME = {
    'hetero': {
        'HeteroIvSelectionCoord': HeteroIvSelectionCoord
    },
    'homo':{
        # TODO
    }
}

class FeatureSelectionCoord(FeatureSelectionBase, CheckDTypeCoord):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, context: Context, 
                model_info: Optional[Dict]=None):        
        """
        Base selection class for both hetero and homo cases.

        Args:
            meta_params, a dict contains params like ks_thres, iv_thres and etc..
        """
        super().__init__(meta_params, context, model_info)
        self.check = CheckDTypeCoord(meta_params)

    @ClassMethodAutoLog()
    def train(self, train_data: Optional[IBondDataFrame]=None, 
                verify_data: Optional[IBondDataFrame]=None, 
                feat_infos: Optional[Dict]=dict()) -> Tuple[Optional[IBondDataFrame]]:
        """
        Base selection interface, common for guest/host/coordinator. It is the main entrance
        for feature selection, for both hetero and homo senario.

        Args:
            None.
        """
        data_attributes = FeatureDataFrame(pd.DataFrame(feat_infos), None)
        self._d_data = self.check.check_d_type()

        for method, config in self.pipeline:
            class_obj = self._class_init(METHOD_NAME, method, "Coord")(self._meta_params, config)
            train_data, data_attribute = class_obj.select_feature(train_data, data_attributes)

        model_infos = self.save_model(data_attributes.to_pandas().to_dict(), [])
        output = make_output(train_data, verify_data)

        return output, model_infos


    @ClassMethodAutoLog()
    def predict(self) -> None:
        pass
