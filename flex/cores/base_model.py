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

from typing import List, Union, Dict, Optional
from logging import getLogger

from .parser import parse_federal_info, parse_sec_param
from .commu_model import CommuModel
from .encrypt_model import EncryptModel
from flex.utils import ClassMethodAutoLog
from flex.crypto.key_exchange.api import make_agreement
from flex.sec_config import DH_KEY_LENGTH


class BaseModel(object):
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List]):
        """
        Inits of federation information fro communication and secure params for security calculation
        ----

        Args:
            federal_info: dict, federal info
            sec_param: dict, params for security calc
        ----

        **Example**:
            >>> federal_info = {
            >>>    "server": "localhost:6001",
            >>>    "session": {
            >>>        "role": "host",
            >>>        "local_id": "zhibang-d-014010",
            >>>        "job_id": 'test_job',
            >>>    },
            >>>    "federation": {
            >>>        "host": ["zhibang-d-014010"],
            >>>        "guest": ["zhibang-d-014011"],
            >>>        "coordinator": ["zhibang-d-014012"]
            >>>    }
            >>> }

            >>> sec_parma = [['paillier', {"key_length": 1024}], ]

            >>> BaseModel(federal_info, sec_param)
        """
        # logger inits
        self.logger = getLogger(self.__class__.__name__)

        # parse federation info
        self.federal_info = parse_federal_info(federal_info)
        # commu inits
        self.commu = CommuModel(federal_info)
        # job_id msg
        self.job_id = self.federal_info.job_id

        # parse secure params
        self.sec_param = parse_sec_param(sec_param)
        self.ecc = []    # save all encrypt method

    @ClassMethodAutoLog()
    def _init_encrypt(self, share_party: Optional[List] = None,
                      local_party: Optional[str] = None) -> None:
        """
        This file mainly init encrypt method
        ----

        Args:
            share_party: list, party ip msg for generate diffle hellman seed
                         if encrypt method
        ----

        Returns:
            None
        ----

        **Example**
        >>> share_party = None
        >>> local_party = None
        >>> self._init_encrypt(share_party, local_party)
        """
        if share_party is None:
            seed = None
        else:
            seed = self._generate_diffie_hellman_seed(share_party, local_party, DH_KEY_LENGTH)

        # encryption inits
        if self.sec_param.empty:
            self.ecc = [EncryptModel()]
        else:
            for i, method in enumerate(self.sec_param.method):
                self.ecc.append(EncryptModel(method, self.sec_param.param[i], seed))

    @property
    def pf_ecc(self) -> EncryptModel:
        return self.ecc[0]

    @ClassMethodAutoLog()
    def _generate_diffie_hellman_seed(self, share_party: List,
                                      local_party: str,
                                      key_length: int = 2048) -> int:
        """
        This method support key exchange in n part
        ----

        Args:
            share_party: list, party generate public seed
            local_party: str,
            key_length: int, key length of Diffie Hellman method
        ----

        Returns:
            seed value, int

        ----
        **Examples**:
            >>> share_party = self.federal_info.guest_host
            >>> local_party = 'zhibang-d-014010'
            >>> key_length = 2048
            >>> seed = self._generate_diffe_hellman_seed(share_party, local_party, key_length)
        """
        # generate
        seed = make_agreement(remote_id=share_party,
                              local_id=local_party,
                              job_id=self.job_id,
                              key_length=key_length
                            )

        return seed

    def save_first_pub_private_key(self) -> Dict:
        """
        Save first public/private key of encrypt method
        ----

        Return:
             public/private key
        """
        key_msg = dict()
        key_msg['pub_key'] = self.pf_ecc.en
        key_msg['pri_key'] = self.pf_ecc.de
        return key_msg

    def over_loading_first_pub_private_key(self, key_msg: Dict) -> None:
        """
        This method mainly load key msg

        Args:
            key_msg: dict, save en/de key messages

        Returns:
             None
        ----

        **Example**:
        >>> method = "paillier"
        >>> param = {"key_length": 1024}
        >>> ecm = EncryptModel(method, param)
        >>> key_msg = dict()
        >>> key_msg['pubkey'] = ecm.en
        >>> key_msg['private_key'] = ecm.de
        >>> BaseModel.over_loading_first_pub_private_key(key_msg)
        """
        if key_msg.get('pub_key') is None or key_msg.get('pri_key') is None:
            raise ValueError('must given encrypt/decrypt object')
        self.pf_ecc.en = key_msg.get('pub_key')
        self.pf_ecc.de = key_msg.get('pri_key')


@ClassMethodAutoLog()
def send_pubkey(channel, encryptor, tag='*') -> None:
    """
    Send encryptor to other party
    """

    channel.broadcast(encryptor, tag=tag)


@ClassMethodAutoLog()
def get_pubkey(channel, tag='*') -> object:
    """
    Get decrypter from certain party
    """

    decrypter = channel.broadcast(tag=tag)
    return decrypter


@ClassMethodAutoLog()
def send_pubkey_signal(channel, encryptor) -> None:
    """
    Send encryptor to signal party
    """

    channel.send(encryptor)


@ClassMethodAutoLog()
def get_pubkey_signal(channel) -> object:
    """
    Get decrypter from certain party
    """

    decrypter = channel.recv()
    return decrypter
