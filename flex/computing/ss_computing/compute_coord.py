"""SS_COMPUTE Protocol
"""
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

from os import path
from typing import Union, Dict, Optional, List

import numpy as np
import torch

from flex.cores.base_model import BaseModel
from flex.cores.check import CheckMixin
from flex.crypto.smpc.smpc_protocol.secure_nn import secureNNTH as SMPC
from flex.crypto.smpc.smpc_protocol.fixed_precision_tensor import make_fixed_precision_tensor, \
    decode_tensor
from flex.constants import DEFAULT_MPC_PRECISION
from flex.utils import ClassMethodAutoLog
from .compute_mixin import Mixin


class SSComputeCoord(BaseModel, Mixin):
    """
    SSCompute Protocol implementation for coordinator

    Args:
        BaseProtocol: base class of protocol
        Mixin: mix in class for all parties

    Returns:
        SSComputeCoord: the SSCompute protocol instance for coordinator
    """

    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        Initlizatte SSComputeCoord class

        Args:
            federal_info: dict, federal_info used in the main Protocol
            sec_param: parameters for crypto
            algo_param: dict: parameters for the mpc
        Returns:
            SSComputeCoord: the SSCompute protocol instance for coordinator
        ----

        **Example:**
        >>> federal_info = {
        >>>    "session": {
        >>>        "role": "coordinator",
        >>>        "local_id": "td-006",
        >>>        "job_id": "test_abc",
        >>>    }
        >>>    "federation": {
        >>>        "host": ["td-005"],
        >>>        "guest": ["td-001"],
        >>>        "coordinator": ["td-006"],
        >>>    }
        >>> }

        >>> sec_param = {
        >>> }

        >>> algo_param = {
        >>>    "mpc_precision": 3,
        >>> }

        """
        # if sec_param is not None:
        #     self.load_default_sec_param(path.join(path.dirname(__file__), 'sec_param.json'))
        # self.load_default_algo_param(path.join(path.dirname(__file__), 'algo_param.json'))
        # super().__init__(federal_info, sec_param, algo_param)
        BaseModel.__init__(self,
                           federal_info=federal_info,
                           sec_param=sec_param)

        # inits encrypt
        self._init_encrypt()

        self.check = CheckMixin

        mpc_conf = self._make_mpc_conf(self.federal_info, sec_param)

        self.smpc = SMPC(federal_info['session']['job_id'], mpc_conf, federal_info)
        self.reverse_mapping = self._mapping_to_mpc(mpc_conf)

        self.chans_broadcast = self.commu.coord_broadcast_channel('broadcast')

    def add(self,
            lhs_sh: Optional[torch.Tensor] = None,
            rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        Addition method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand of additon in secret share
            rhs_sh: Union[torch.tensor, None], right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        pass

    def add_rec(self,
                lhs_sh: Optional[torch.Tensor] = None,
                rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        Addition and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand of additon in secret share
            rhs_sh: Union[torch.tensor, None], right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        self.smpc.tag += 1

    def substract(self,
                  lhs_sh: Optional[torch.Tensor] = None,
                  rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        Substract method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand of additon in secret share
            rhs_sh: Union[torch.tensor, None], right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        pass

    def substract_rec(self,
                      lhs_sh: Optional[torch.Tensor] = None,
                      rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        Substract and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand of additon in secret share
            rhs_sh: Union[torch.tensor, None], right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        self.smpc.tag += 1

    def mul(self,
            lhs_sh: Optional[torch.Tensor] = None,
            rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        Substract method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None

        """
        return self.smpc.mul(lhs_sh, rhs_sh)

    def mul_rec(self,
                lhs_sh: Optional[torch.Tensor] = None,
                rhs_sh: Optional[torch.Tensor] = None,
                pre: Optional[str] = 'default') -> None:
        """
        Mul and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None

        """
        self.mul(lhs_sh, rhs_sh)
        self.smpc.tag += 1
        return None

    def matmul(self,
               lhs_sh: Optional[torch.Tensor] = None,
               rhs_sh: Optional[torch.Tensor] = None):
        """
        Substract method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None

        """
        return self.smpc.matmul(lhs_sh, rhs_sh)

    def matmul_rec(self,
                   lhs_sh: Optional[torch.Tensor] = None,
                   rhs_sh: Optional[torch.Tensor] = None,
                   pre: Optional[str] = 'default') -> None:
        """
        Mul and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None

        """
        self.matmul(lhs_sh, rhs_sh)
        self.smpc.tag += 1
        return None

    def ge0(self, lhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        ge0 compare method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], operand in secret share

        Return:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        # Step 1: gather shape of operands from host and guest
        shape_guest, shape_host = self.chans_broadcast.gather(tag='ge0shape' + str(self.smpc.tag))
        assert shape_guest == shape_host
        # Step 2: Making auxalilay matrix for coordinator aided compare
        lhs_sh = torch.zeros(shape_guest, dtype=torch.long)
        # Step 3: Call the underline MPC compare protocol
        self.smpc.relu_deriv(lhs_sh)

    def ge0_rec(self, lhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        ge0 compare and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], operand in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        self.ge0(lhs_sh)
        self.smpc.tag += 1

    def ge(self,
           lhs_sh: Optional[torch.Tensor] = None,
           rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        ge compare method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand operand in secret share
            lhs_sh: Union[torch.tensor, None], right hand operand in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        self.ge0(None)

    def ge_rec(self,
               lhs_sh: Optional[torch.Tensor] = None,
               rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        ge compare and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand operand in secret share
            lhs_sh: Union[torch.tensor, None], right hand operand in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        self.ge(lhs_sh, rhs_sh)
        self.smpc.tag += 1

    def le(self,
           lhs_sh: Optional[torch.Tensor] = None,
           rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        le compare method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand operand in secret share
            lhs_sh: Union[torch.tensor, None], right hand operand in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        self.ge0(None)

    def le_rec(self,
               lhs_sh: Optional[torch.Tensor] = None,
               rhs_sh: Optional[torch.Tensor] = None) -> None:
        """
        ge compare and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: Union[torch.tensor, None], left hand operand in secret share
            lhs_sh: Union[torch.tensor, None], right hand operand in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = None
        >>> rhs_sh = None
        >>> None
        """
        self.le(lhs_sh, rhs_sh)
        self.smpc.tag += 1

    def share_secrets(self,
                      secret: Optional[torch.Tensor] = None,
                      pre: Optional[str] = None) -> None:
        """
        share secrets to secret shares method for SSCompute Protocol

        Args:
            secret: Union[torch.tensor, None], operand to be considered

        Returns:
            None
        ----

        *Example:**
        >>> secret = None
        >>> None
        """
        return self.smpc.share_secrets(secret)

    def share_secret_from(self,
                          secret: Optional[torch.Tensor] = None,
                          pre: Optional[str] = None) -> None:
        """
        share secrets to init secret shares for SSCompute Protocol

        Args:
            secret: torch.tensor, data to be shared
            src: the party_id who will share

        Returns:
            secret_sh

        """
        return None
