from typing import Dict, List, Optional, Tuple, Union

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from .common import HeteroIvSelectionBase


class HeteroIvSelectionParticipant(HeteroIvSelectionBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Optional[Dict]=dict()):\
        super().__init__(meta_params, config)

    @ClassMethodAutoLog()
    def select_feature(self, data: IBondDataFrame, \
                            data_attributes: FeatureDataFrame) -> \
                            Tuple[IBondDataFrame, FeatureDataFrame]:          
        """
        Hetero-iv participant selection.

        Args:
           data: IBondDataFrame, input data.
           data_attributes: data attributes, should include iv for each feature.

        Return:
            IBondDataFrame, output data.
            FeatureDataFrame, updated data attributes.
        """
        bin_info = data_attributes.bin_info
        data_attributes.update('iv', self.iv_thres, self.down_feature_num)
        if self.top_k is not None:
            data_attributes = data_attributes.iloc[:self.top_k]

        data_attributes.update_bin(bin_info)
        data = data[data_attributes.feature_cols]

        return data, data_attributes