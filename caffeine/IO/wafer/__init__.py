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
#  File name: __init__.py
#  Created by liwei on 2020/12/17.
#  Edited by zhiqiang on 2021/1/25.

from typing import Dict, Union, List
from logging import getLogger
from enum import Enum, unique

from pydantic import BaseModel

from .dataframe.ibond_dataframe import IBondDataFrame
from .session.ibond_conf import IBondConf
from .session.session_factory import SessionFactory
from .util.decorator import singleton


@unique
class TableTypeEnum(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class InputODM(BaseModel):
    source: Dict[str, str] = {}
    column_mapping: Dict[str, List[str]] = {}
    outputs: Dict = {}


class MiddlewareODM(BaseModel):
    engine: str = "light"
    warehouse: str = "/model"
    bulletin: Dict = {}
    report_warehouse: str = "/report"
    data_warehouse: str = "/data"


@singleton
class Wafer(object):
    """
    This Wafer class is an interface used by fl algorithm library caffeine.
    """

    def __init__(self, config: Dict[str, Union[str, Dict]]):
        """
        Init class by configuartion.py

        Args:
            config: Dict, must be agreed with MiddlewareODM.

        Example:
        >>> config = {
                      "engine": "light",
                      "inputs": {
                                 "train_data": "train.parquet",
                                 "val_data": "val.parquet"
                                }
                     }
        >>> Wafer(config)
        """
        self.config = MiddlewareODM.parse_obj(config)
        self.session = (
            SessionFactory.builder.app_name("wafer_session")
            .config(conf=IBondConf(config=self.config.dict()))
            .get_or_create()
        )
        self.logger = getLogger(self.__class__.__name__)

    # def get_inputs(self, inputs: Dict) -> Dict[str, IBondDataFrame]:
    #    """
    #    This function reads all input data and return IBondDataFrame according to config.
    #    Args:

    #    Returns:
    #        Dict[str, IBondDataFrame], for example {"train_data": IBondDataFrame, "val_data": IBondDataFrame}

    #    Exception:
    #        IOError，when data source is unavailable or unable to parse, an IOError will be raised.

    #    Example:
    #    >>> config = {
    #            "inputs": {
    #                "source": {
    #                    "train_data": "dataset/train_guest.parquet",
    #                    "val_data": "dataset/test_guest.parquet"
    #                },
    #                "column_mapping": {
    #                    "id": [
    #                        "ID"
    #                    ],
    #                    "y": [
    #                        "Label"
    #                    ]
    #                }
    #            }
    #                 }
    #    >>> wafer = Wafer(config)
    #    >>> Model.train(**wafer.get_inputs())
    #    """
    #    result = {}
    #    for name, uri in getattr(inputs, "source").items():
    #        try:
    #            df = self.session.read(uri, inputs.get("column_mapping"))
    #        except Exception:
    #            self.logger.exception("Loading data failed!!!")
    #            raise IOError

    #        result[name] = df

    #    return result
