from typing import Dict, List, Optional, Tuple, Union

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from .common import HeteroIvSelectionBase


class HeteroIvSelectionCoord(HeteroIvSelectionBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Optional[Dict]=dict()):\
        super().__init__(meta_params, config)

    @ClassMethodAutoLog()
    def select_feature(self, data: IBondDataFrame, \
                            data_attributes: FeatureDataFrame) -> \
                            None:          
        """
        Hetero-ks coordinator ensemble selection, get broadcast message only.

        Args:
            No input.
        """
        return data, data_attributes