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
from typing import Union, Dict, List, Optional

import numpy as np
import torch

from flex.crypto.onetime_pad.api import generate_onetime_pad_encryptor
from flex.cores.base_model import BaseModel
from flex.utils import ClassMethodAutoLog
from flex.cores.iterative_apply import iterative_divide
from flex.crypto.onetime_pad.iterative_add import iterative_add


class NNBaseModel(BaseModel):
    """
    NN base model to init communication channel and secure params
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        NN gradinent update protocol param inits
        inits of federation information for communication and secure params for security calculation

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

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

        >>> sec_param = [['onetime_pad', {'key_length': 512}], ]

        >>> algo_param = {}

        >>> OTPNNFTGuest(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        # inits communication
        self.grad_channel = self.commu.guest2host_broadcast_channel('grad')


class OTPNNFTGuest(NNBaseModel):
    """
    NN gradient update protocol, Guest side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        NNBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

    @ClassMethodAutoLog()
    def exchange(self, grads: Union[list, np.ndarray, torch.Tensor],
                 label: Union[list, np.ndarray, torch.Tensor],
                 *args, **kwargs) -> Union[list, np.ndarray, torch.Tensor]:
        """
        This method mainly update model's gradient in one item

        Args:
            grads: local gradient
            label: labels msg of guest

        Return:
            The average gradient
        ----

        **Example:**
        >>> grads = [[torch.Tensor(2, 3).uniform_(-1, 1)], [torch.Tensor(3, 2).uniform_(-1, 1)]]
        >>> label = np.random.randint(0, 2, (10,))
        >>> self.exchange(grads, label)
        """
        # Get host local gradient
        enc_grads = self.grad_channel.gather()
        self.logger.info('Guest get host local gradient')

        # Use label to select the correct gradient
        selected_grads = []
        for i in range(len(enc_grads)):
            for j in range(len(label)):
                selected_grads.append(enc_grads[i][label[j]][j])
        self.logger.info('Guest use label to select the correct gradient')

        # Calculate the mean gradient and send it to host
        host_sum_grads = sum(selected_grads[i] for i in range(len(selected_grads))).decode()
        host_avg_grads = iterative_divide(host_sum_grads, len(label))
        sum_grads = iterative_add(grads, host_avg_grads)
        avg_grads = iterative_divide(sum_grads, len(enc_grads) + 1)
        self.grad_channel.broadcast(avg_grads)
        self.logger.info('Guest calculate the mean gradient and send it to host')

        return avg_grads


class OTPNNFTHost(NNBaseModel):
    """
    NN gradient update protocol, Host side
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict] = None):
        NNBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

    @ClassMethodAutoLog()
    def exchange(self, grads: Union[list, np.ndarray, torch.Tensor],
                 *args, **kwargs) -> Union[list, np.ndarray, torch.Tensor]:
        """
        This method mainly update model's gradient in one item

        Args:
            grads: local gradient

        Returns:
            The average gradient
        ----

        **Example:**
        >>> grads = [[torch.Tensor(2, 3).uniform_(-1, 1)], [torch.Tensor(3, 2).uniform_(-1, 1)]]
        >>> self.exchange(grads)
        """
        # generate a encryptor list to mask local gradient
        entropy = secrets.randbits(512)
        enc_grads = []
        encryptors = []
        num_labels = len(grads)
        batch_size = len(grads[0])
        [encryptors.append(generate_onetime_pad_encryptor(entropy)) for i in range(len(grads) * 2)]

        # padding the gradient
        for i in range(num_labels):
            enc_grads.append([encryptors[i].encrypt(grads[i][j], alpha=1) if j != batch_size - 1 else encryptors[
                num_labels + i].encrypt(grads[i][j], alpha=-1) for j in range(batch_size)])
        for i in range(num_labels):
            for j in range(batch_size - 2):
                enc_grads[i][batch_size - 2].ciphertext = encryptors[num_labels + i].encrypt(
                    enc_grads[i][batch_size - 2].ciphertext, alpha=-1).ciphertext
        self.logger.info('host padding the gradient')

        # send encrypted gradient to guest and receive the mean gradient
        self.grad_channel.gather(enc_grads)
        avg_grads = self.grad_channel.broadcast()
        self.logger.info('send encrypted gradient to guest and receive the mean gradient')

        return avg_grads
