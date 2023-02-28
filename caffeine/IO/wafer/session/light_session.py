#!/usr/bin/python3
#
#  _____________             _________   ___       __      ________
#  ___(_)__  __ )__________________  /   __ |     / /_____ ___  __/____________
#  __  /__  __  |  __ \_  __ \  __  /    __ | /| / /_  __ `/_  /_ _  _ \_  ___/
#  _  / _  /_/ // /_/ /  / / / /_/ /     __ |/ |/ / / /_/ /_  __/ /  __/  /
#  /_/  /_____/ \____//_/ /_/\__,_/      ____/|__/  \__,_/ /_/    \___//_/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond Wafer
#  File name: light_dataframe.py
#  Created by liwei on 2020/12/8.


import json
import os
from typing import Optional, Dict, List, Union, Tuple, Any
from pathlib import Path
import time
import uuid
from abc import ABCMeta

import pandas as pd

from .ibond_session import IBondSession
from .ibond_conf import IBondConf
from ..dataframe.light_dataframe import LightDataFrame
from ..dataframe.light_sparse_dataframe import LightSparseDataFrame
#from ..util.decorator import singleton
from ..util.decorator import Singleton
from ..util.xlog import logger
from ..util.http import post

MODEL_HANDLERS = {}


class ABCSingleton(ABCMeta, Singleton):
    pass


class ModelHandler:
    MODEL_DIR = "model"
    MODELIO_DIR = "modelio"

    def __init__(
        self,
        conf: IBondConf,
        model_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.conf = conf
        dag_task_uuid = self.conf.bulletin.bond_dag_task_uuid

        if "." in model_name:
            self.operator_name, self.model_name = model_name.rsplit(".", 1)
        else:
            self.model_name = model_name
            self.operator_name = self.conf.bulletin.name
        self.model_id = model_id

        self.modelio_path = Path(
            os.path.join(
                self.conf.warehouse, self.MODELIO_DIR, dag_task_uuid, self.operator_name
            )
        )

        Path.mkdir(self.modelio_path, parents=True, exist_ok=True)

        def increase():
            n = 0
            while True:
                yield n
                n = n + 1

        self._it = increase()

    @property
    def model_idx(self):
        return next(self._it)

    def save_model(
        self,
        model: Dict,
    ) -> Tuple[Union[int, Any], Union[Path, Any]]:
        model_idx = self.model_idx
        random_id = uuid.uuid1().hex
        model_id = f'{random_id}_{model_idx}'
        path = self.modelio_path / f'{self.model_name}_{model_id}'

        with open(path, "w") as f:
            f.write(json.dumps(model))

        return model_id, str(path)

    def load_model(
        self,
        model_idx: Optional[str],
        model_path: Optional[str],
    ) -> Dict:
        """
        Load model by model index.
        model storage:
            1. dag train flow
                /modelio/{dag_task_uuid}/{operator_name}_{output_name}_idx
            2. model package
                /model/{model_package_uuid}/{operator_name}_{output_name}_idx

        Args:
            model_package_uuid, str, the model id of the model to load.

        Returns:
            bytes, model bson bytes

        Example:
        >>> model_json = load_model(model_package_uuid)
        """
        model_package_uuid = getattr(self.conf.bulletin, "model_package_uuid", None)
        if model_path is None:
            if model_package_uuid is not None:
                # search model
                model_package_dir = Path(
                    os.path.join(self.conf.warehouse, self.MODEL_DIR, model_package_uuid)
                )
                model_files = [f.name for f in model_package_dir.rglob("*.model")]
                model_file = self.get_closest_model_file(
                    self.operator_name, model_files
                )
                model_path = os.path.join(
                    self.conf.warehouse, self.MODEL_DIR, model_package_uuid, f"{model_file}"
                )
                logger.info(f"ready to load modelpackage model: {model_path}")
            elif model_idx is not None:
                model_path = self.modelio_path / f"{self.model_name}_{model_idx}"
            else:
                raise RuntimeError("fail to load model")
        with open(model_path, "r") as f:
            b = f.read()
        return json.loads(b)

    def get_closest_model_file(self, operator_name, files):
        min_distance = -1
        target = None
        for f in files:
            distance = self.minDistance(operator_name + ".model", f)
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                target = f
        return target

    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        # 有一个字符串为空串
        if n * m == 0:
            return n + m
        # DP 数组
        D = [[0] * (m + 1) for _ in range(n + 1)]
        # 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j
        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)
        return D[n][m]


#@singleton
class LightSession(IBondSession, metaclass=ABCSingleton):
    DATASET_DIR = "dataset"
    DATAIO_DIR = "dataio"
    ROC_TABLE_PRE = "roc_curve"
    KS_TABLE_PRE = "ks_curve"
    DETAIL_TABLE_PRE = "details"

    def __init__(self, *args, **kwargs):
        self.metric_func = {"roc": self._roc, "ks": self._ks, "detail": self._detail}
        super().__init__(*args, **kwargs)

    def create_dataframe(
        self, pdf: pd.DataFrame, data_desc: Optional[Dict[str, List]] = None,
        feat_desc: Optional[Dict] = None, to_sparse: bool = False
    ):
        if not to_sparse:
            return LightDataFrame(pdf, data_desc, feat_desc, self.conf)
        else:
            return LightSparseDataFrame(pdf, data_desc, feat_desc, self.conf)

    def read(
        self, uri: Union[str, Path], data_desc: Dict[str, List] = None,
        feat_desc: Dict = None
    ) -> LightDataFrame:
        """
        Decide how to load data automatically according to uri content.
        Args:
            uri: str, A path to file. Currently supports 3 types: "*.parquet", "*.csv",
                table name
            data_desc: definition of [id, y]
        Returns:
            LightDataFrame

        Raise:
            TypeError
        """
        if isinstance(uri, Path):
            uri_path = uri
        elif isinstance(uri, str):
            uri_path = Path(uri)
        else:
            raise TypeError(f"uri type {type(uri)} is not supported.")

        if uri_path.suffix == ".parquet":
            read_func = self.read_parquet
        elif uri_path.suffix == ".csv":
            read_func = self.read_csv
        elif hasattr(self.conf, "host") and hasattr(self.conf, "port"):
            read_func = self.read_mysql
            uri_path = uri
        else:
            raise TypeError(f"Can not decide data source type for uri={uri_path}.")

        return read_func(uri_path, data_desc, feat_desc)

    def read_csv(
        self, uri: Union[str, Path], data_desc: Optional[Dict[str, List]] = None,
        feat_desc: Optional[Dict] = None
    ) -> LightDataFrame:
        pdf = pd.read_csv(uri)
        return LightDataFrame(pdf, data_desc, feat_desc, self.conf)

    def read_parquet(
        self, uri: Union[str, Path], data_desc: Optional[Dict[str, List]] = None,
        feat_desc: Optional[Dict] = None
    ) -> LightDataFrame:
        pdf = pd.read_parquet(uri)
        return LightDataFrame(pdf, data_desc, feat_desc, self.conf)

    def read_mysql(
        self, table_name: str, data_desc: Optional[Dict[str, List]] = None,
        feat_desc: Optional[Dict] = None
    ) -> LightDataFrame:
        pdf = pd.read_sql_table(
            table_name,
            f"mysql://{self.conf.host}:{self.conf.port}/{self.conf.db_name}?user={self.conf.user_name}&password={self.conf.password}",  # noqa: E501
        )
        return LightDataFrame(pdf, data_desc, feat_desc, self.conf)

    def read_table(
        self, table_name: str, data_desc: Optional[Dict[str, List]] = None,
        feat_desc: Optional[Dict] = None
    ) -> LightDataFrame:
        path = f"{self.conf.warehouse}/{self.DATASET_DIR}/{table_name}"
        return self.read_parquet(path, data_desc, feat_desc)

    def read_tmp_table(
        self, table_name: str, data_desc: Optional[Dict[str, List]] = None,
        feat_desc: Optional[Dict] = None
    ) -> LightDataFrame:
        operator_name, _table_name = table_name.split(".")
        path = os.path.join(
            self.conf.warehouse,
            self.DATAIO_DIR,
            self.dag_task_uuid,
            operator_name,
            _table_name,
        )
        return self.read_parquet(path, data_desc, feat_desc)

    def calc_table_size(self, table_name: str) -> int:
        path = os.path.join(self.conf.warehouse, self.DATASET_DIR, table_name)

        if not os.path.isdir(path):
            return os.path.getsize(path)

        files = os.listdir(path)

        _sum = 0
        for file in files:
            _sum += os.path.getsize(os.path.join(path, file))
        return _sum

    def _extract(self, metrics, keys):
        curve = metrics["curve"]
        data = {key: curve.get(key) for key in keys}
        table = pd.DataFrame.from_dict(data)
        return table

    def _roc(self, metrics):
        keys = ["threshold", "tpr", "fpr", "auc_m"]
        return self._extract(metrics, keys)

    def _ks(self, metrics):
        keys = ["threshold", "tpr", "fpr", "ks"]
        return self._extract(metrics, keys)

    def _detail(self, metrics):
        table = pd.DataFrame(
            columns=(
                "accuracy",
                "precision",
                "recall",
                "tp",
                "fp",
                "tn",
                "fn",
                "f1score",
            )
        )
        table = table.append(metrics["detail_table"], ignore_index=True)
        return table

    def save_metrics(self, metric_type: str, table_name: str, metrics: Dict, save_loc:Optional[str] = "dataio"):
        if metric_type not in self.metric_func:
            raise RuntimeError(f"metric_type invalid: {metric_type}")

        func = self.metric_func[metric_type]
        table = func(metrics)

        ibond_table = self.create_dataframe(table)
        if save_loc == "dataio":
            ibond_table.save_as_tmp_table(table_name)
        elif save_loc == "dataset":
            ibond_table.save_as_table(table_name)
        else:
            raise TypeError(f"illegal param for save_loc=\"{save_loc}\".")

    def _save_metric_table(self, metrics: Dict, epoch: int):
        roc_table_name = (
            f"{self.ROC_TABLE_PRE}_{self.conf.bulletin['bond_dag_task_uuid']}_{epoch}"
        )
        ks_table_name = (
            f"{self.KS_TABLE_PRE}_{self.conf.bulletin['bond_dag_task_uuid']}_{epoch}"
        )
        detail_table_name = f"{self.DETAIL_TABLE_PRE}_{self.conf.bulletin['bond_dag_task_uuid']}_{epoch}"  # noqa: E501

        # roc
        self.save_metrics("roc", roc_table_name, metrics, "dataset")
        # ks
        self.save_metrics("ks", ks_table_name, metrics, "dataset")
        # detail
        self.save_metrics("detail", detail_table_name, metrics, "dataset")

        return {"roc": roc_table_name, "ks": ks_table_name, "detail": detail_table_name}

    def _register_model(self, model_info):
        if any(
            (
                "report_url" not in self.conf.bulletin,
                "bond_monitor_id" not in self.conf.bulletin,
                "bond_dag_task_id" not in self.conf.bulletin,
            )
        ):
            logger.warning("skip register model due to lack of bulletin information")
            return

        info = model_info.copy()
        info["dag_task_id"] = self.conf.bulletin["bond_dag_task_id"]
        info["description"] = ""
        info["verify_result"] = ""
        info["generate_time"] = int(time.time())

        if info["metrics"]:
            verify_result = self._save_metric_table(
                info["metrics"], info["epoch_times"]
            )
            info["verify_result"] = json.dumps(verify_result)
            del info["metrics"]
        # swap algo_name and model_type due to the old version implement mistake.
        # Here we still keep it.
        info["algo_name"], info["model_type"] = info["model_type"], info["algo_name"]
        logger.info("model_info:" + str(info))
        url = self.conf.bulletin["report_url"] + "/api/model/manager/save"

        try:
            res = post(url, json.dumps(info))
        except Exception as e:
            logger.error("model save register connection failed: " + repr(e))
            return
        logger.info("register model done: " + url)
        logger.info("register model response: " + str(res))

    def _get_model_handler(self, model_name):
        model_handler = MODEL_HANDLERS.get(model_name)
        if model_handler is None:
            if hasattr(self.conf.bulletin, "model_id"):
                model_handler = ModelHandler(
                    self.conf, model_name, self.conf.bulletin.model_id
                )
            else:
                model_handler = ModelHandler(self.conf, model_name, None)
        MODEL_HANDLERS.setdefault(model_name, model_handler)
        return model_handler

    def save_model(
        self,
        model: Dict,
        model_info: Optional[Dict] = None,
        model_name: Optional[str] = "model",
    ) -> str:
        model_idx, path = self._get_model_handler(model_name).save_model(model)
        if model_info is not None:
            model_info["generate_time"] = int(time.time())
            model_info["model_path"] = path
            model_info["version"] = str(model_idx)

            self._register_model(model_info)

        return model_idx

    def load_model(
        self,
        model_idx: Optional[str] = 0,
        model_path: Optional[str] = None,
        model_name: Optional[str] = "model",
    ) -> Dict:
        return self._get_model_handler(model_name).load_model(model_idx, model_path)

    def load_model_from_modelpackage(self, model_name) -> Dict:
        return self._get_model_handler(model_name).load_model(None, None)

    def load_model_by_path(self, model_path):
        with open(model_path, "r") as f:
            b = f.read()
        return json.loads(b)
        return
