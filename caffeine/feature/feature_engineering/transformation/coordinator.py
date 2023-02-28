from typing import Dict, Tuple, Optional

import pandas as pd

from .common import FeatureTransformBase
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.utils import make_output
from .binning.hetero_binning.dtbinning.coordinator import HeteroDtBinCoord
from .mapping.hetero_mapping.woe_mapping.coordinator import HeteroMapToWOECoord
from .mapping.local_mapping.map_to_int import MapToInt
from .base_transform import TransformBase


TRANSFORM_METHODS =  {
    'hetero': {
        'dt_bin': HeteroDtBinCoord,
        'woe_map': HeteroMapToWOECoord
    },
    'homo':{
        # TODO
    },
    'local': {
        'equifrequent_bin': TransformBase,
        'equidist_bin': TransformBase,
        'integer_map': TransformBase,
        'onehot_map': TransformBase
    }
}


class FeatureTransformCoord(FeatureTransformBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, context: Context, 
                        model_info: Optional[Dict]=None):
        super().__init__(meta_params, context, model_info)

    @ClassMethodAutoLog()
    def train(self, train_data: Optional[IBondDataFrame]=None, 
                verify_data: Optional[IBondDataFrame]=None, 
                feat_infos: Optional[Dict]=dict()) -> Tuple:
        """
        Training method for training data.

        Args:
            data: IBondDataFrame, training data.
        """
        data_attributes = FeatureDataFrame(pd.DataFrame(feat_infos), None)

        if self.pipeline is not None:
            for trans_process, config in self.pipeline:
                trans_processer = TRANSFORM_METHODS[self.process_method][trans_process]
                method = trans_processer(self._meta_params, config, self._context)
                train_data, data_attributes = method.fit(train_data, data_attributes)

        model_infos = self._report_and_models(data_attributes)
        output = make_output(train_data, verify_data, data_attributes.to_pandas().to_dict())

        return output, model_infos

    @ClassMethodAutoLog()
    def predict(self) -> None:
        """
        Test method for test data.

        Args:
            data: IBondDataFrame, training data.

        Return:
            data: IBondDataFrame, training data after transformation.
        """
        pass