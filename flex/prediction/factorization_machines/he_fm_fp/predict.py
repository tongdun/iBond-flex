#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from typing import Dict, List, Optional

import numpy as np

from flex.utils import ClassMethodAutoLog
from flex.training.factorization_machines.he_fm_ft.mixin import FMMixin
from flex.cores.check import CheckMixin
from flex.cores.base_model import BaseModel
from flex.algo_config import HE_FM_FT_PARAM


class HEFMFPBaseModel(BaseModel):
    """
    FM prediction base model to init communication channel and secure params
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        """
        FM prediction protocol param inits
        inits of federation information for communication and secure params for security calculation
        Args:
            federal_info: dict, federal info
            sec_param: dict, params for security calc
            algo_param: dict, params for algo

        -----

        **Example:**

        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "guest",
        >>>        "local_id": "zhibang-d-014011",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> sec_param = []

        >>> algo_param = {}

        >>> HEFMFTGuest(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        # inits communication
        self.data_channel = self.commu.guest2host_broadcast_channel('date')

        # data type check
        self.check = CheckMixin

    @ClassMethodAutoLog()
    def _check_input(self, theta: np.ndarray,
                     v: np.ndarray,
                     features: np.ndarray):
        # input data type check
        self.check.array_type_check(theta)
        self.check.array_type_check(v)
        self.check.array_type_check(features)

        # data dimension check
        self.check.data_dimension_check(theta, 1)
        self.check.data_dimension_check(v, 2)
        self.check.data_dimension_check(features, 2)

        # check related data dimension
        self.check.data_relation_check(theta.shape[0], features.shape[1])
        self.check.data_relation_check(v.shape[0], features.shape[1])


class HEFMFPGuest(HEFMFPBaseModel, FMMixin):
    """
    FM gradient update protocol, Guest side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        HEFMFPBaseModel.__init__(self,
                                 federal_info=federal_info,
                                 sec_param=sec_param,
                                 algo_param=algo_param)

    @ClassMethodAutoLog()
    def exchange(self, theta: np.ndarray,
                 v: np.ndarray, features: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        """
        This method mainly get prediction in one item
        Args:
            theta: weight params of FM model's linear term
            v: weight params of FM model's embedding term
            features: origin dataset
        Return:
            prediction of this item

        -----

        **Example:**

        >>> theta = np.array([0.2, 0.3, 0.5])
        >>> v = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> features = np.array([[2, 4, 1], [3, 6, 1], [5, 7, 2], [6, 4, 9]])
        >>> self.exchange(theta, v, features)
        """
        # check input data
        self._check_input(theta, v, features)
        self.logger.info('guest complete input data check')

        # get F(x) and embedding term
        guest_f, guest_embedding = FMMixin.fm(theta, v, features)
        host_f_embedding = self.data_channel.gather()
        self.logger.info('guest get host F(x) and embedding term')

        # if host more than one, calculation sum of F(x) and embedding term
        host_f = []
        host_embedding = []
        for i in range(len(host_f_embedding)):
            host_f.append(host_f_embedding[i][0])
            host_embedding.append(host_f_embedding[i][1])
        host_f = np.sum(host_f, axis=0)
        host_embedding = np.sum(host_embedding, axis=0)

        # calc FM model prediction
        forward = guest_f + host_f + np.sum(guest_embedding * host_embedding, axis=1)
        clip_param = HE_FM_FT_PARAM['clip_param']
        forward = np.clip(forward, -1 * clip_param, clip_param)
        predict = 1. / (1. + np.exp(-forward))

        return predict


class HEFMFPHost(HEFMFPBaseModel, FMMixin):
    """
    FM  prediction protocol, Host side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        HEFMFPBaseModel.__init__(self,
                                 federal_info=federal_info,
                                 sec_param=sec_param,
                                 algo_param=algo_param)

    @ClassMethodAutoLog()
    def exchange(self, theta: np.ndarray,
                 v: np.ndarray, features: np.ndarray,
                 *args, **kwargs) -> None:
        """
        This method mainly get prediction in one item
        Args:
            theta: weight params of FM model's linear term
            v: weight params of FM model's embedding term
            features: origin dataset
        Return:
            None

        -----

        **Example:**

        >>> theta = np.array([0.2, 0.3])
        >>> v = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> features = np.array([[2, 4], [3, 6], [5, 7], [6, 4]])
        >>> self.exchange(theta, v, features)
        """
        # data shape check
        self._check_input(theta, v, features)
        self.logger.info('host complete input data check')

        # send host F(x) and embedding term
        host_f, host_embedding = FMMixin.fm(theta, v, features)
        self.data_channel.gather([host_f, host_embedding])
        self.logger.info('host send F(x) and embedding term')

        return
