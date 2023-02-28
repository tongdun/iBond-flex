"""
 Created by liwei on 2020/12/8.
"""
from typing import List, Tuple, Union, Optional, Dict
from logging import getLogger
from abc import ABCMeta, abstractmethod
import pandas as pd

from ..session.ibond_conf import IBondConf


class IBondDataFrame(metaclass=ABCMeta):
    def __init__(
        self,
        pdf,
        data_desc: Dict[str, List] = None,
        feat_desc: Dict = None,
        conf: IBondConf = None,
    ):
        self._data_desc = {
            "id_desc": [],
            "y_desc": [],
            "time": [],
            "pred": [],
            "local_pred": [],
            "other": [],
        }
        self._feat_desc = dict()
        if data_desc is not None:
            self._data_desc.update(data_desc)
        if feat_desc is not None:
            self._feat_desc.update(feat_desc)
        self.conf = conf
        self._hooks = []
        self._context = {}

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __getitem__(self, x: Union[List[str], str]) -> "IBondDataFrame":
        pass

    @abstractmethod
    def select(self) -> "IBondDataFrame":
        pass

    @abstractmethod
    def save_as_table(self):
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def toSeries(self, index: Optional[str]) -> pd.Series:
        pass

    @abstractmethod
    def drop(self, columns):
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def join(self, other):
        pass

    @abstractmethod
    def batches(self, batch_size: int, drop_last: bool = False):
        pass

    @abstractmethod
    def to_local_iterator(self):
        pass

    @abstractmethod
    def shuffle(self):
        pass

    @property
    @abstractmethod
    def data_desc(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def feat_desc(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def prediction(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def columns(self):
        raise NotImplementedError()
