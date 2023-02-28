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
from typing import Union, Dict, Optional, List

import numpy as np
import torch

from flex.cores.base_model import BaseModel
from flex.cores.check import CheckMixin
from flex.crypto.smpc.smpc_protocol.secure_nn import secureNNTH as SMPC
from flex.crypto.smpc.smpc_protocol.fixed_precision_tensor import make_fixed_precision_tensor, \
    decode_tensor
from flex.utils import ClassMethodAutoLog
from flex.constants import DEFAULT_MPC_PRECISION
from .compute_mixin import Mixin


class SSComputeHost(BaseModel, Mixin):
    """SSCompute Protocol implementation for host
        Args:
            BaseProtocol: base class of protocol
            Mixin: mix in class for all parties
        Returns:
            SSComputeHost: the SSCompute protocol instance for host
    """

    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        Initlizatte SSComputeHost class

        Args:
            federal_info: dict, federal_info used in the main Protocol
            sec_param: parameters for crypto
            algo_param: dict: parameters for the mpc

        Returns:
            SSComputeHost: the SSCompute protocol instance for host
        ----

        **Example:**
        >>> federal_info = {
        >>>    "session": {
        >>>        "role": "host",
        >>>        "local_id": "td-005",
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

        self.var_mapping = dict()

    def add(self, lhs_sh: torch.Tensor, rhs_sh: torch.Tensor) -> torch.Tensor:
        """
        Addition method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            ret_sh = (lhs_sh + rhs_sh) % 2**64
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([-2872629282,28262829,29728363])
        >>> torch.tensor([-2872600044, 35890211, -2924563904])
        """
        return lhs_sh + rhs_sh

    def add_rec(self,
                lhs_sh: torch.Tensor,
                rhs_sh: torch.Tensor,
                pre: Optional[str] = 'default') -> Union[torch.Tensor, np.ndarray]:
        """
        Addition and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            ret: torch.tensor, result of add and reconstruct
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([-2872629282,28262829,29728363])
        >>> torch.tensor([2, 3, 4])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([2872629284,-28262826,-29728360])
        >>> torch.tensor([2, 3, 4])
        """
        ret_sh = self.add(lhs_sh, rhs_sh)
        ret = self.smpc.reconstruct(ret_sh)
        return self.post_process_data(ret, pre)

    def substract(self, lhs_sh: torch.Tensor, rhs_sh: torch.Tensor) -> torch.Tensor:
        """
        Substract method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([2872629282,-28262829,-29728363])
        >>> torch.tensor([-2872600044, 35890211, -2924563904])
        """
        return lhs_sh - rhs_sh

    def substract_rec(self,
                      lhs_sh: torch.Tensor,
                      rhs_sh: torch.Tensor,
                      pre: Optional[str] = 'default') -> Union[torch.Tensor, np.ndarray]:
        """
        Substract and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([2872629282,-28262829,-29728363])
        >>> torch.tensor([2, 3, 4])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([-2872629284,28262826,29728360])
        >>> torch.tensor([2, 3, 4])
        """
        ret_sh = self.substract(lhs_sh, rhs_sh)
        ret = self.smpc.reconstruct(ret_sh)
        return self.post_process_data(ret, pre)

    def mul(self, lhs_sh: torch.Tensor, rhs_sh: torch.Tensor) -> torch.Tensor:
        """
        Substract method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([2872629282,-28262829,-29728363])
        >>> torch.tensor([-2872600044, 35890211, -2924563904])
        """
        return self.smpc.mul(lhs_sh, rhs_sh)

    def mul_rec(self,
                lhs_sh: torch.Tensor,
                rhs_sh: torch.Tensor,
                pre: Optional[str] = 'default') -> Union[torch.Tensor, np.ndarray]:
        """
        Mul and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([2872629282,-28262829,-29728363])
        >>> torch.tensor([2, 3, 4])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([-2872629284,28262826,29728360])
        >>> torch.tensor([2, 3, 4])
        """
        ret_sh = self.mul(lhs_sh, rhs_sh)
        ret = self.smpc.reconstruct(ret_sh)
        if self.smpc.precision > 0:
            ret = self.smpc._cutoff(ret, 2 ** 64 // 10 ** self.smpc.precision)
        return self.post_process_data(ret, pre)

    def matmul(self, lhs_sh: torch.Tensor, rhs_sh: torch.Tensor) -> torch.Tensor:
        """
        Substract method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([2872629282,-28262829,-29728363])
        >>> torch.tensor([-2872600044, 35890211, -2924563904])
        """
        return self.smpc.matmul(lhs_sh, rhs_sh)

    def matmul_rec(self,
                   lhs_sh: torch.Tensor,
                   rhs_sh: torch.Tensor,
                   pre: Optional[str] = 'default') -> Union[torch.Tensor, np.ndarray]:
        """
        Mul and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627382, -29272292267])
        >>> rhs_sh = torch.tensor([2872629282,-28262829,-29728363])
        >>> torch.tensor([2, 3, 4])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([-2872629284,28262826,29728360])
        >>> torch.tensor([2, 3, 4])
        """
        ret_sh = self.matmul(lhs_sh, rhs_sh)
        ret = self.smpc.reconstruct(ret_sh)
        if self.smpc.precision > 0:
            ret = self.smpc._cutoff(ret, 2 ** 64 // 10 ** self.smpc.precision)
        return self.post_process_data(ret, pre)

    def ge0(self, lhs_sh: torch.Tensor) -> torch.Tensor:
        """
        Compare ge0 (>=0) method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, operand in secret share

        Returns:
            ret_sh
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29239, 7627382, -29272292269])
        >>> torch.tensor([-2383262682902, 3223982792837, 923872398724])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> torch.tensor([2383262682903, -3223982792836, -923872398724])
        """
        self.chans_broadcast.gather(lhs_sh.shape, tag='ge0shape' + str(self.smpc.tag))
        return self.smpc.relu_deriv(lhs_sh)

    def ge0_rec(self,
                lhs_sh: torch.Tensor,
                pre: Optional[str] = 'default') -> Union[torch.Tensor, np.ndarray]:
        """
        Compare ge0 (>=0) and then reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, operand in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29239, 7627382, -29272292269])
        torch.tensor([1, 1, -1])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        torch.tensor([1, 1, -1])
        """
        ret_sh = self.ge0(lhs_sh)
        ret = self.smpc.reconstruct(ret_sh)
        return self.post_process_data(ret, pre, False)

    def ge(self, lhs_sh: torch.Tensor, rhs_sh: torch.Tensor) -> torch.Tensor:
        """
        Ge (>=) method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627383, -29272292267])
        >>> rhs_sh = torch.tensor([2872629284,-28262826,-29728363])
        >>> torch.tensor([229876239151233, -39986123465, 491234756192987])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([-2872629282,28262826,29728360])
        >>> torch.tensor([-229876239151233, 39986123465, -491234756192986])
        """
        return self.ge0(lhs_sh - rhs_sh)

    def ge_rec(self,
               lhs_sh: torch.Tensor,
               rhs_sh: torch.Tensor,
               pre: Optional[str] = 'default') -> Union[torch.Tensor, np.ndarray]:
        """
        Ge (>=) and reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627383, -29272292267])
        >>> rhs_sh = torch.tensor([2872629284,-28262826,-29728363])
        >>> torch.tensor([0, 1, 0])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([-2872629282,28262826,29728360])
        >>> torch.tensor([0, 1, 0])
        """
        ret_sh = self.ge(lhs_sh, rhs_sh)
        ret = self.smpc.reconstruct(ret_sh)
        return self.post_process_data(ret, pre, False)

    def le(self, lhs_sh: torch.Tensor, rhs_sh: torch.Tensor) -> torch.Tensor:
        """
        Le (>=) method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627383, -29272292267])
        >>> rhs_sh = torch.tensor([2872629284,-28262826,-29728363])
        >>> torch.tensor([229876239151234, -39986123465, 491234756192986])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([-2872629282,28262826,29728360])
        >>> torch.tensor([-229876239151233, 39986123466, -491234756192986])
        """
        return self.ge0(rhs_sh - lhs_sh)

    def le_rec(self,
               lhs_sh: torch.Tensor,
               rhs_sh: torch.Tensor,
               pre: Optional[str] = 'default') -> Union[torch.Tensor, np.ndarray]:
        """
        Le (>=) and reconstruct method for SSCompute Protocol

        Args:
            lhs_sh: torch.tensor, left hand of additon in secret share
            rhs_sh: torch.tensor, right hand of additon in secret share

        Returns:
            None
        ----

        **Example:**
        >>> lhs_sh = torch.tensor([29238, 7627383, -29272292267])
        >>> rhs_sh = torch.tensor([2872629284,-28262826,-29728363])
        >>> torch.tensor([1, 0, 1])

        (on the other party)
        >>> lhs_sh = torch.tensor([-29238, -7627382, 29272292268])
        >>> rhs_sh = torch.tensor([-2872629282,28262826,29728360])
        >>> torch.tensor([1, 0, 1])
        """
        ret_sh = self.le(lhs_sh, rhs_sh)
        ret = self.smpc.reconstruct(ret_sh)
        return self.post_process_data(ret, pre, False)

    def share_raw(self,
                  raw) -> List:
        """
        share raw to init secret shares for SSCompute Protocol

        Args:
            raw: any, data to be shared

        Returns:
            [raw_0, raw_1, ..., raw_N]
        ----

        **Example:**
        >>> raw = torch.tensor([2, 3, -5])
        >>> [torch.tensor([2, 3, -5]),
        >>>  torch.tensor([7, -23, -712])]

        (on the other party)
        >>> secret = torch.tensor([7, -23, -712])
        >>> [torch.tensor([2, 3, -5]),
        >>>  torch.tensor([7, -23, -712])]
        """
        return self.smpc.share_raw(raw)

    def share_secrets(self,
                      secret: Union[torch.Tensor, np.ndarray],
                      pre: Optional[str] = 'default') -> List[torch.Tensor]:
        """
        share secrets to init secret shares for SSCompute Protocol

        Args:
            secret: torch.tensor, data to be shared

        Returns:
            [secret_sh_0, secret_sh_1, ..., secret_sh_N]
        ----

        **Example:**
        >>> secret = torch.tensor([2, 3, -5])
        >>> [torch.tensor([-10523948710236, 986723561234, 21329487234978]),
        >>>  torch.tensor([2987236239872, -23984365123487, -71234769213867])]

        (on the other party)
        >>> secret = torch.tensor([1, -2, 3])
        >>> [torch.tensor([10523948710238, -986723561231, -21329487234983]),
        >>>  torch.tensor([-2987236239871, 23984365123485, 71234769213870])]
        """
        secret = self.pre_process_data(secret, pre)
        return self.smpc.share_secrets(secret)

    def share_secret_from(self,
                          secret: Union[torch.Tensor, np.ndarray],
                          src: torch.Tensor, pre: Optional[str] = 'default') -> torch.Tensor:
        """
        share secrets to init secret shares for SSCompute Protocol

        Args:
            secret: torch.tensor, data to be shared
            src: the party_id who will share

        Returns:
            secret_sh

        """
        secret = self.pre_process_data(secret, pre)
        src_mpc = self.reverse_mapping[src]
        return self.smpc.share_secret_from(secret, src_mpc)

    def pre_process_data(self,
                         vec: Union[torch.Tensor, np.ndarray],
                         name_vec: Optional[str] = 'default') -> torch.Tensor:
        """
        convert vec usable by SS_COMPUTE

        Args:
            vec: Union[np.ndarray, torch.tensor]: raw data
            name_vec: string, name of the vec

        Returns:
            vec_encoded: torch.longtensor: able to run SS_COMPUTE
        """
        vec, in_type, in_dtype = self._check_input(vec)
        self.var_mapping[name_vec] = (in_type, in_dtype)

        return make_fixed_precision_tensor(vec, self.smpc.precision)

    def post_process_data(self,
                          vec: torch.Tensor,
                          name_vec: Optional[str] = 'default',
                          scale: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        convert vec to original format

        Args:
            vec: torch.tensor, smpc result
            name_vec: string, name of the vec

        Returns:
            vec_decoded: Union[np.ndarray, torch.tensor], result in original format
        """
        in_type, in_dtype = self.var_mapping[name_vec]
        fix_precision = self.smpc.precision if scale else 0

        return decode_tensor(vec, fix_precision, in_type, in_dtype)
