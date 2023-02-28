from typing import Optional, Union, Dict, List, Tuple

import pandas as pd
from scipy import sparse

from .light_dataframe import LightDataFrame
from ..session.ibond_conf import IBondConf
from ..util.xlog import logger


class LightSparseDataFrame(LightDataFrame):
    def __init__(
        self,
        pdf: pd.DataFrame,
        data_desc: Dict[str, List] = None,
        feat_desc: Dict = None,
        conf: IBondConf = None,
    ):
        """
        params: pdf - pandas sparse dataframe
        """
        if not pdf.dtypes.apply(pd.api.types.is_sparse).any(): 
            logger.warn('dataframe is not sparse, force converting to sparse dataframe ...')
            pdf = pd.DataFrame.sparse.from_spmatrix(sparse.coo_matrix(pdf.values), columns=pdf.columns)
        super().__init__(pdf, data_desc, feat_desc, conf)

    def derive(
        self, pdf: pd.DataFrame, data_desc: Dict[str, List] = None,
        feat_desc: Dict = None
    ) -> "LightSparseDataFrame":
        if data_desc is None:
            data_desc = self._data_desc
        if feat_desc is None:
            feat_desc = self._feat_desc

        return LightSparseDataFrame(pdf, data_desc, feat_desc, self.conf)


    def to_dense(self) -> "LightDataFrame":
        self._pdf = self._pdf.sparse.to_dense()
        return super().derive(self._pdf, self.data_desc, self._feat_desc)
