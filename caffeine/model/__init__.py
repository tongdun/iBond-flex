#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: __init__.py
#
#  Create date: 2020/11/24
#
from logging import getLogger
from typing import Union

from .base_model import AbstractModel
from .logistic_regression.cross_feature.with_coord.coordinator import \
    HeteroLogisticRegressionCoord
from .logistic_regression.cross_feature.with_coord.guest import \
    HeteroLogisticRegressionGuest
from .logistic_regression.cross_feature.with_coord.host import \
    HeteroLogisticRegressionHost


class MockModel(AbstractModel):
    def __init__(self, meta_param):
        self.logger = getLogger(self.__class__.__name__)
        self.logger.info(f"Init MockModel using meta_param={meta_param}")

    def train(self, train_data, val_data):
        self.logger.info(
            f"Training MockModel using train_data={train_data}, val_data={val_data}")

    def predict(self, data):
        self.logger.info(f"Predicting MockModel using data={data}")

    def _save_params(self):
        self.logger.info(f"Saveing MockModel.")

    def load_params(self, params):
        self.logger.info(f"Loading MockModel using params={params}")

    def feature_selection(self):
        self.logger.info("in MockModel.feature_selection")
        
        


modules = {
    "HeteroLogisticRegression": {
        "host": HeteroLogisticRegressionHost,
        "guest": HeteroLogisticRegressionGuest,
        "coordinator": HeteroLogisticRegressionCoord
    },
    "MockModel": {
        "host": MockModel,
        "guest": MockModel,
        "coordinator": MockModel
    }

}


def has_module(name: str) -> bool:
    return (name in modules)


def get_module(name: str, role: str = None) -> Union[AbstractModel, None]:
    return modules.get(name, {}).get(role, None)
