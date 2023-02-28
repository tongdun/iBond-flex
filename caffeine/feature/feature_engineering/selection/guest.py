from typing import Dict, List, Optional, Tuple
from logging import getLogger

import pandas as pd
import json

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.utils.dataframe import parse_ibonddf
from caffeine.feature.feature_engineering.check_d_type import CheckDTypeGuest
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.utils import make_output
from caffeine.feature.feature_engineering.selection.hetero_selection.hetero_iv.participant import HeteroIvSelectionParticipant
from .common import FeatureSelectionBase


METHOD_NAME = {
    'hetero': {
        'HeteroIvSelectionGuest': HeteroIvSelectionParticipant
    },
    'homo':{
         # TODO
    },
    'local':{
        # TODO
    }
}

class FeatureSelectionGuest(FeatureSelectionBase, CheckDTypeGuest):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, context: Context, 
                model_info: Optional[Dict]=None):        
        """
        Base selection class for both hetero and homo cases.

        Args:
            meta_params, a dict contains params like ks_thres, iv_thres and etc..
        """
        super().__init__(meta_params, context, model_info)
        self.check = CheckDTypeGuest(meta_params)

    @ClassMethodAutoLog()
    def train(self, train_data: Optional[IBondDataFrame], 
                verify_data: Optional[IBondDataFrame], 
                feat_infos: Optional[Dict]) -> Tuple:
        """
        Guest train selection interface. It is the main entrance
        for feature selection, for both hetero and homo senario.

        Args:
            input_data: IBondDataFrame, training data.
            verify_data: IBondDataFrame, verify data.

        Return:
            dict, with output datasets and model.
            list, a list of models.
        """
        data_attributes = FeatureDataFrame(pd.DataFrame(feat_infos), train_data.data_desc)
        self._d_data = self.check.check_d_type(data_attributes.shape)

        for method, config in self.pipeline:
            class_obj = self._class_init(METHOD_NAME, method, "Guest")(self._meta_params, config)
            self.logger.info(f'>>>> data_attributes {data_attributes}')
            train_data, data_attributes = class_obj.select_feature(train_data, data_attributes)

        if verify_data is not None:
            verify_data = verify_data[data_attributes.feature_cols]
        
        output = make_output(train_data, verify_data)

        model_infos = self._report_and_models(data_attributes)

        return output, model_infos


    @ClassMethodAutoLog()
    def predict(self, data: IBondDataFrame) -> IBondDataFrame:
        """
        Guest test selection interface. It is the main entrance
        for feature selection, for both hetero and homo senario.

        Args:
            input: IBondDataFrame, input data.
            
        Returns:
            IBondDataFrame, updated data.
        """
        data_desc = parse_ibonddf(data)
        data = data[data_desc['id_cols'] + data_desc['y_cols'] +  self.selected_x_cols]
        return data

