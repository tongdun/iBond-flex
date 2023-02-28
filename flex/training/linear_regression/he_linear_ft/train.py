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
from typing import Dict, Optional, List

import numpy as np

from flex.constants import *
from flex.cores.base_model import BaseModel, send_pubkey, get_pubkey
from flex.cores.check import CheckMixin
from flex.utils import ClassMethodAutoLog


class LinearBaseModel(BaseModel):
    """
    Linear regression base model to init communication channel and secure params
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        Linear regression calculation loss protocol param inits
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

        >>> algo_param = { }

        >>> HELinearFTGuest(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)
        # inits encrypt
        self._init_encrypt()

        # inits channel
        self.key_channel = self.commu.coord_broadcast_channel('pub_key')
        self.pred_channel = self.commu.coord_broadcast_channel('pred')
        self.loss_channel = self.commu.coord_broadcast_channel('loss')

        # data type check
        self.check = CheckMixin

    @ClassMethodAutoLog()
    def _check_input(self, local_pred: np.ndarray) -> None:
        # input data type check
        self.check.array_type_check(local_pred)

        # data dimension check
        self.check.data_dimension_check(local_pred, 1)

    def _do_exchange(self, delta: np.ndarray,
                     *args, **kwargs) -> np.ndarray:
        """
        The part that host and guest have in common to get loss

        Args:
            delta: intermediate result to get loss

        Returns:
            loss of this item, to update gradient
        -----

        **Example:**
        >>> prng = np.random.RandomState(0)
        >>> delta = np.array(prng.uniform(-1, 1, (8,)))
        """
        # encrypt the intermediate result
        ciphertext = self.pf_ecc.encrypt(delta)
        self.pred_channel.gather(ciphertext)
        self.logger.info('guest and host send encrypted grad to coord')

        # get loss from coord
        loss = self.loss_channel.broadcast()
        self.logger.info('get loss from coord')

        return loss


class HELinearFTCoord(LinearBaseModel):
    """
    Linear regression calculation loss protocol
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        LinearBaseModel.__init__(self,
                                 federal_info=federal_info,
                                 sec_param=sec_param,
                                 algo_param=algo_param)

        # send pubkey host
        send_pubkey(self.key_channel, self.pf_ecc.en)

    @ClassMethodAutoLog()
    def exchange(self, *args, **kwargs):
        # get the encrypted gradients
        ciphertexts = self.pred_channel.gather()
        self.logger.info('coord get the encrypted gradients')

        # calculation loss and decrypt it
        loss = self.pf_ecc.decrypt(sum(ciphertexts))

        # send loss to host and guest
        self.loss_channel.broadcast(loss)
        self.logger.info('coord send loss to host and guest')

        return


class HELinearFTGuest(LinearBaseModel):
    """
    Linear regression calculation loss protocol
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        LinearBaseModel.__init__(self,
                                 federal_info=federal_info,
                                 sec_param=sec_param,
                                 algo_param=algo_param)

        # get pubkey from coord
        self.pf_ecc.en = get_pubkey(self.key_channel)

    @ClassMethodAutoLog()
    def _check_input(self, local_pred: np.ndarray, label: np.ndarray):
        super()._check_input(local_pred=local_pred)

        # input data type check
        self.check.array_type_check(label)

        # data dimension check
        self.check.data_dimension_check(label, 1)

    @ClassMethodAutoLog()
    def exchange(self, local_pred: np.ndarray, label: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        """
        This method mainly calculation loss in one item

        Args:
            local_pred: local prediction
            label: labels msg of guest

        Returns:
            loss of this item, to update gradient
        -----

        **Example:**
        >>> prng = np.random.RandomState(0)
        >>> local_pred = np.array(prng.uniform(-1, 1, (8,)))
        >>> label = np.array(prng.randint(0, 2, (8,)))
        >>> trainer = make_protocol(HE_LINEAR_FT, federal_info, sec_param, algo_param=None)
        >>> result = trainer.exchange(local_pred, label)
        """
        # check input data
        self._check_input(local_pred, label)

        # calculation the intermediate result for get loss
        delta = local_pred - label

        # calculation loss
        loss = self._do_exchange(delta)

        return loss


class HELinearFTHost(LinearBaseModel):
    """
    Linear regression calculation loss protocol
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        LinearBaseModel.__init__(self,
                                 federal_info=federal_info,
                                 sec_param=sec_param,
                                 algo_param=algo_param)

        # get pubkey from coord
        self.pf_ecc.en = get_pubkey(self.key_channel)

    @ClassMethodAutoLog()
    def exchange(self, local_pred: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        """
        This method mainly calculation loss in one item

        Args:
            local_pred: local prediction

        Returns:
            loss of this item, to update gradient
        -----

        **Example:**
        >>> prng = np.random.RandomState(0)
        >>> grad = np.array(prng.uniform(-1, 1, (8,)))
        >>> trainer = make_protocol(HE_LINEAR_FT, federal_info, sec_param, algo_param=None)
        >>> result = trainer.exchange(grad)
        """
        # check input data
        self._check_input(local_pred)

        # calculation loss
        loss = self._do_exchange(local_pred)

        return loss
