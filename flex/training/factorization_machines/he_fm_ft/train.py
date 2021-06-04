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
from typing import Tuple, Dict, List, Optional

import numpy as np

from flex.utils import ClassMethodAutoLog
from flex.training.factorization_machines.he_fm_ft.mixin import FMMixin
from flex.cores.check import CheckMixin
from flex.cores.base_model import BaseModel
from flex.algo_config import HE_FM_FT_PARAM


class HEFMBaseModel(BaseModel):
    """
    FM base model to init communication channel and secure params
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        """
        FM gradinent update protocol param inits
        inits of federation information for communication and secure params for security calculation
        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
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

        >>> sec_param = [['paillier', {"key_length": 1024}], ]

        >>> algo_param = {}

        >>> FMFTGuest(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        # inits communication
        self.data_channel = self.commu.guest2host_broadcast_channel('date')
        self.label_channel = self.commu.guest2host_broadcast_channel('label')

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


class HEFMFTGuest(HEFMBaseModel, FMMixin):
    """
    FM gradient update protocol, Guest side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        HEFMBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

    @ClassMethodAutoLog()
    def _check_input(self, theta: np.ndarray,
                     v: np.ndarray,
                     features: np.ndarray,
                     labels: np.ndarray):
        super()._check_input(theta=theta, v=v, features=features)

        # input data type check
        self.check.array_type_check(labels)

        # data dimension check
        self.check.data_dimension_check(labels, 1)

        # check related data dimension
        self.check.data_relation_check(features.shape[0], labels.shape[0])

    @ClassMethodAutoLog()
    def exchange(self, theta: np.ndarray,
                 v: np.ndarray, features: np.ndarray,
                 labels: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method mainly update model's gradient in one item

        Args:
            theta: weight params of FM model's linear term
            v: weight params of FM model's embedding term
            features: origin dataset
            labels: labels msg of guest

        Returns:
            gradient of theta and v update
        -----

        **Example:**
        >>> theta = np.array([0.2, 0.3, 0.5])
        >>> v = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> features = np.array([[2, 4, 1], [3, 6, 1], [5, 7, 2], [6, 4, 9]])
        >>> labels = np.array([[0], [1], [1], [0]])
        >>> self.exchange(theta, v, features, labels)
        """
        # check input data
        self._check_input(theta, v, features, labels)
        self.logger.info('guest complete input data check')

        # get F(x) and embedding term
        guest_f, guest_embedding = FMMixin.fm(theta, v, features)
        self_embedding = guest_embedding
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

        # send common items to host for update gradient
        diff_y = 0.25 * forward - 0.5 * labels
        sum_embedding = host_embedding + guest_embedding
        self.label_channel.broadcast([diff_y, sum_embedding])
        self.logger.info('guest send common items to host for update gradient')

        # update gradient theta
        grad_theta = np.mean((diff_y.reshape(diff_y.shape[0], 1) * features), axis=0)

        # get the v gradient
        v_update = np.zeros(self_embedding.shape, dtype=np.float32)
        for i in range(features.shape[0]):
            v_update[i, :] = diff_y[i] * (sum_embedding - self_embedding)[i, :]
        grad_v = np.mean(v_update, axis=0)

        return grad_theta, grad_v


class HEFMFTHost(HEFMBaseModel, FMMixin):
    """
    FM gradient update protocol, Host side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        HEFMBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)

    @ClassMethodAutoLog()
    def exchange(self, theta: np.ndarray,
                 v: np.ndarray, features: np.ndarray,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method mainly update model's gradient in one item

        Args:
            theta: weight params of FM model's linear term
            v: weight params of FM model's embedding term
            features: origin dataset

        Returns:
            gradient of theta and v update
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
        self_embedding = host_embedding
        self.data_channel.gather([host_f, host_embedding])
        self.logger.info('host send F(x) and embedding term')

        # get common items for update gradient
        diff_y_sum_embedding = self.label_channel.broadcast()
        diff_y = diff_y_sum_embedding[0]
        sum_embedding = diff_y_sum_embedding[1]
        self.logger.info('host get common items for update gradient')

        # update gradient theta
        grad_theta = np.mean((diff_y.reshape(diff_y.shape[0], 1) * features), axis=0)

        # get the v gradient
        v_update = np.zeros(self_embedding.shape, dtype=np.float32)
        for i in range(features.shape[0]):
            v_update[i, :] = diff_y[i] * (sum_embedding - self_embedding)[i, :]
        grad_v = np.mean(v_update, axis=0)

        return grad_theta, grad_v
