"""
 Created by liwei on 2020/12/8.
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
from caffeine.IO.wafer.util.report import Reporter

from .ibond_conf import IBondConf
from ..dataframe.ibond_dataframe import IBondDataFrame


class IBondSession(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        conf: IBondConf,
    ):
        self.conf = conf

        bulletin = getattr(self.conf, "bulletin", None)
        self.dag_task_uuid = getattr(bulletin, "bond_dag_task_uuid", None)
        report_url = getattr(bulletin, "report_url", None)
        bond_monitor_id = getattr(bulletin, "bond_monitor_id", None)
        self.reporter = Reporter(report_url, bond_monitor_id)

    def report(self, key, val_str):
        return self.reporter.insert(key, val_str)

    @abstractmethod
    def create_dataframe(
        self, pdf: pd.DataFrame, data_desc: Optional[Dict[str, List]] = None,
        to_sparse: bool = False
    ) -> IBondDataFrame:
        pass

    @abstractmethod
    def read(
        self, uri: str, data_desc: Optional[Dict[str, List]] = None
    ) -> IBondDataFrame:
        pass

    @abstractmethod
    def read_csv(
        self, uri: str, data_desc: Optional[Dict[str, List]] = None
    ) -> IBondDataFrame:
        pass

    @abstractmethod
    def read_parquet(
        self, uri: str, data_desc: Optional[Dict[str, List]] = None
    ) -> IBondDataFrame:
        pass

    @abstractmethod
    def read_mysql(
        self, table_name: str, data_desc: Optional[Dict[str, List]] = None
    ) -> IBondDataFrame:
        pass

    @abstractmethod
    def read_table(
        self, table_name: str, data_desc: Optional[Dict[str, List]] = None
    ) -> IBondDataFrame:
        pass

    @abstractmethod
    def calc_table_size(self, table_name: str):
        pass

    @abstractmethod
    def save_model(
        self, model: Dict, model_info: Optional[Dict], model_name: str
    ) -> str:
        pass

    @abstractmethod
    def load_model(
        self,
        model_idx: Optional[str],
        model_path: Optional[str],
        model_name: Optional[str],
    ) -> Dict:
        pass
