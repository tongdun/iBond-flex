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

import secrets
from typing import Tuple, Dict, Optional, List

import numpy as np

from flex.cores.base_model import BaseModel, send_pubkey, get_pubkey
from flex.cores.check import CheckMixin
from flex.utils import ClassMethodAutoLog
from flex.utils.activations import sigmoid


class LRBaseModel(BaseModel):
    """
        Logistic regression(No coordinate)  base model
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        LR gradinent update protocol param inits
        inits of federation information for communication and secure params for security calculation

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example**:
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

        >>> algo_param = { }

        >>> LRBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()

        # inits channel
        self.data_channel = self.commu.guest2host_broadcast_channel('data')
        self.label_channel = self.commu.guest2host_broadcast_channel('label')
        self.grad_channel = self.commu.guest2host_broadcast_channel('grad')
        self.key_channel = self.commu.guest2host_broadcast_channel('pub_key')

        # data type check
        self.check = CheckMixin

    @ClassMethodAutoLog()
    def _check_input(self, theta: np.ndarray,
                     features: np.ndarray):
        # input data type check
        self.check.array_type_check(theta)
        self.check.array_type_check(features)

        # data dimension check
        self.check.data_dimension_check(theta, 1)
        self.check.data_dimension_check(features, 2)

        # check related data dimension
        self.check.data_relation_check(theta.shape[0], features.shape[1])


class HEOTPLR1Guest(LRBaseModel):
    """
    LR gradient update protocol
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        LRBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

        # send pubkey host
        send_pubkey(self.key_channel, self.pf_ecc.en)

    @ClassMethodAutoLog()
    def _check_input(self, theta: np.ndarray,
                     features: np.ndarray, labels: np.ndarray):
        super()._check_input(theta=theta, features=features)

        # input data type check
        self.check.array_type_check(labels)

        # data dimension check
        self.check.data_dimension_check(labels, 1)

        # check related data dimension
        self.check.data_relation_check(features.shape[0], labels.shape[0])

    @ClassMethodAutoLog()
    def exchange(self, theta: np.ndarray,
                 features: np.ndarray, labels: np.ndarray,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method mainly update model's gradient in one item

        Args:
            theta: weight params of LR model
            features: origin dataset
            labels: labels msg of guest

        Return:
            prediction of this item, gradient of theta update
        ----

        **Example:**
        >>> theta = np.array([0.2, 0.3, 0.5])
        >>> features = np.array([
        >>> [2, 4, 1], [3, 6, 1], [5, 7, 2], [6, 4, 9]])
        >>> labels = np.array([[0], [1], [1], [0]])
        >>> HEOTPLR1Guest.exchange(theta, features, labels)
        """
        # check input data
        self._check_input(theta, features, labels)
        self.logger.info('guest complete input data check')

        # get prediction of host part
        u2 = self.data_channel.gather()
        self.logger.info('guest get model prediction of host part')

        # calc LR model prediction
        u1 = theta.dot(features.T)
        u = u1 + sum(u2)
        h_x = sigmoid(u)

        # send encrypt label-y_predict to host
        enc_diff_y = self.pf_ecc.encrypt(labels - h_x)
        self.label_channel.broadcast(enc_diff_y)
        self.logger.info('guest send encrypt data of label-prediction')

        # get host decrypt gradient, decrypt gradient then send to host
        enc_padded_grads = self.grad_channel.gather()
        padded_grads = [self.pf_ecc.decrypt(grad) for grad in enc_padded_grads]
        self.grad_channel.scatter(padded_grads)
        self.logger.info('guest send decrypt mask gradient msg')

        # gradient update
        batch_size = features.shape[0]
        grads = (-1 / batch_size) * ((labels - h_x).dot(features))

        return h_x, grads


class HEOTPLR1Host(LRBaseModel):
    """
    LR gradient update protocol
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        LRBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

        # send pubkey host
        self.pf_ecc.en = get_pubkey(self.key_channel)

    @ClassMethodAutoLog()
    def exchange(self, theta: np.ndarray,
                 features: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        """
        This method mainly update model's gradient in one item

        Args:
            theta: weight params of LR model
            features: origin dataset

        Returns:
            prediction of this item, gradient of theta update
        ----

        **Example:**
        >>> theta = np.array([0.2, 0.4])
        >>> features = np.array([
        >>> [3, 4], [8, 6], [5, 9], [7, 3]])
        >>> HEOTPLR1Host.exchange(theta, features)
        """
        # data shape check
        self._check_input(theta, features)
        self.logger.info('host complete input data check')

        # calc model prediction of host part
        u2 = theta.dot(features.T)
        self.data_channel.gather(u2)
        self.logger.info('host send model prediction')

        # get encrypt label msg from guest
        enc_diff_y = self.label_channel.broadcast()
        batch_size = features.shape[0]
        enc_grads = (-1 / batch_size) * (enc_diff_y.dot(features))
        self.logger.info('host get encrypt data of label-prediction')

        # gradient add random mask, send to guest
        rand = secrets.SystemRandom()
        r = np.array([rand.randint(-2 ** 64, 2 ** 64) / 2 ** 48 for i in range(len(enc_grads))])
        enc_r = self.pf_ecc.encrypt(r)
        enc_padded_grads = enc_grads + enc_r
        self.grad_channel.gather(enc_padded_grads)
        self.logger.info('host send encrypt mask gradient msg')

        # gen encrypt gradient msg and minus random mask
        padded_grads = self.grad_channel.scatter()
        grads = padded_grads - r

        return grads
