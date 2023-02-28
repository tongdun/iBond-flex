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
#  File name: moonshot_dataframe.py

import json
import os
import uuid
from typing import Optional, Dict, List, Union, Tuple, Any
from pathlib import Path
import time

import pandas as pd

from .ibond_conf import IBondConf
from ..dataframe.light_dataframe import LightDataFrame
from ..util.decorator import Singleton
from ..util.xlog import logger
from ..util.http import post
from caffeine.IO.wafer.util.report import FileReporter
from .light_session import LightSession

MOONSHOT_MODEL_HANDLERS = {}


class MoonshotModelHandler:
    MODEL_DIR = "model"
    MODELIO_DIR = "modelio"

    def __init__(
            self,
            conf: IBondConf,
    ):
        self.conf = conf
        dag_task_uuid = self.conf.bulletin.bond_dag_task_uuid
        self.operator_name = self.conf.bulletin.name
        self.modelio_path = Path(
            os.path.join(
                self.conf.warehouse, self.MODELIO_DIR, dag_task_uuid
            )
        )
        Path.mkdir(self.modelio_path, parents=True, exist_ok=True)

    def save_model(
            self,
            model: Dict,
    ) -> Tuple[Union[int, Any], Union[Path, Any]]:
        model_id = uuid.uuid1()
        path = self.modelio_path / f"{model_id}"

        with open(path, "w") as f:
            f.write(json.dumps(model))
        return str(model_id), str(path)

    def load_model(
            self,
            model_id: Optional[str],
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
        >>> model_json = load_model()
        """
        if model_path is None:
            if model_id is not None:
                model_path = self.modelio_path / f"{model_id}"
            else:
                raise RuntimeError("fail to load model and model path is Empty")
        if not os.path.exists(model_path):
            raise RuntimeError(f"fail to load model and model path {model_path} is not exist")
        with open(model_path, "r") as f:
            b = f.read()
        return json.loads(b)


class MoonshotSession(LightSession, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.conf.getAll())
        self.reporter = FileReporter(
            os.path.join(
                self.conf.report_warehouse, self.conf.bulletin.bond_dag_task_uuid
            )
        )

    def report(self, key, val_str):
        return self.reporter.insert(key, val_str)

    def save_model(
            self,
            model: Dict,
            model_info: Optional[Dict] = None,
            model_name: Optional[str] = None,
    ) -> str:
        model_id, path = self._get_model_handler().save_model(model)
        if model_info is not None:
            model_info["generate_time"] = int(time.time())
            model_info["model_path"] = path
            model_info["id"] = str(model_id)

            self._register_model(model_info)

        return model_id

    def _get_model_handler(self):
        model_handler = MoonshotModelHandler(self.conf)
        return model_handler

    def load_model(
            self,
            model_idx: Optional[str] = '',
            model_path: Optional[str] = None,
            model_name: Optional[str] = None,
    ) -> Dict:
        return self._get_model_handler().load_model(model_idx, model_path)
