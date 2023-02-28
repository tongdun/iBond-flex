from typing import Dict, Tuple, Optional

import json
import pandas as pd

from .common import FeatureTransformBase
from .binning.local_binning.equidist_bin import EquidistBinning
from .binning.local_binning.equifrequent_bin import EquifrequentBinning
from .binning.hetero_binning.dtbinning.host import HeteroDtBinHost
from .mapping.hetero_mapping.woe_mapping.host import HeteroMapToWOEHost
from .mapping.local_mapping.map_to_int import MapToInt
from .mapping.local_mapping.onehot_map import OneHotMap
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.feature_engineering.utils import make_output
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context


TRANSFORM_METHODS =  {
    'hetero': {
        'dt_bin': HeteroDtBinHost,
        'woe_map': HeteroMapToWOEHost
    },
    'homo':{
        # TODO
    },
    'local': {
        'equifrequent_bin': EquifrequentBinning,
        'equidist_bin': EquidistBinning,
        'integer_map': MapToInt,
        'onehot_map': OneHotMap
    }
}


class FeatureTransformHost(FeatureTransformBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, context: Context, 
                        model_info: Optional[Dict]=None):
        super().__init__(meta_params, context, model_info)


    @ClassMethodAutoLog()
    def train(self, train_data: Optional[IBondDataFrame], 
                verify_data: Optional[IBondDataFrame], 
                feat_infos: Optional[Dict]) -> Tuple:
        """
        Training method for training data.

        Args:
            data: IBondDataFrame, training data.
            verify_data: IBondDataFrame, verify data.

        Return:
            dict, with output datasets.
        """
        data_attributes = FeatureDataFrame(pd.DataFrame(feat_infos), train_data.data_desc)
        
        if self.pipeline is not None:
            for trans_process, config in self.pipeline:
                trans_processer = TRANSFORM_METHODS[self.process_method][trans_process]
                method = trans_processer(self._meta_params, config, self._context)
                train_data, data_attributes = method.fit(train_data, data_attributes)
                self.logger.info(f'>>>>> data {train_data.to_pandas().head()}')

        # save model
        model_infos = self._report_and_models(data_attributes)
        
        # deal with verify_data
        if verify_data is not None:
            verify_data = self.predict(verify_data)
        output = make_output(train_data, verify_data, data_attributes.to_pandas().to_dict())

        return output, model_infos

    @ClassMethodAutoLog()
    def predict(self, data: IBondDataFrame) -> IBondDataFrame:
        """
        Test method for test data.

        Args:
            data: IBondDataFrame, training data.

        Return:
            data: IBondDataFrame, training data after transformation.
        """
        if self.pipeline is not None:
            for test_process, config in self.pipeline:
                test_processer = TRANSFORM_METHODS[self.process_method][test_process](self._meta_params, 
                                        config, self._context, self._model_info['feature_transform'])
                data = test_processer.fit_transform(data)
        
        return data





