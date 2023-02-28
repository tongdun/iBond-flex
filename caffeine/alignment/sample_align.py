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
#                                                                                              
#  File name: sample_align.py                                                                          
#                                                                                              
#  Create date: 2020/11/26                                                                               
#
from flex.api import make_protocol
from flex.constants import SAL, ECDH_SAL, RSA_SAL
from logging import getLogger
from typing import List, Tuple, Union, Optional

from caffeine.utils import ClassMethodAutoLog
from caffeine.utils.config import security_config

align_protocol_map = {
    'sal': SAL,
    'ecdh_sal': ECDH_SAL,
    'rsa_sal': RSA_SAL,
}


class SampleAligner(object):
    """
    Securely align samples.
    """

    def __init__(self,
                 protocol_name: str,
                 federal_info: dict,
                 security: Optional[List] = None):
        """
        Args:
            protocol_name: str, alignment protocol method, supported methods:
                - 'sal' for arbiter assisted secure alignment
                - 'ecdh_sal' for ECDH secure alignment
                - ... more alignment protocol coming soon
            federal_info: dict, dict to describe the federation info.
            security: optional list, security parameters for the alignment
                protocol, if None, will use default settings in config.

        -----   

        **Examples:**

        >>> protocol_name = 'sal'
        >>> federal_info = {
            "server": "localhost:6001",
            "session": {
                "role": "host",
                "local_id": "zhibang-d-014010",
                "job_id": 'test_job',
            },
            "federation": {
                "host": ["zhibang-d-014010"],
                "guest": ["zhibang-d-014011"],
                "coordinator": ["zhibang-d-014012"]
            }
        }
        >>> aligner = SampleAligner(protocol_name, federal_info)
        """
        self.logger = getLogger(self.__class__.__name__)
        # NOTE dirty transform for compatibility
        protocol_name = align_protocol_map.get(protocol_name, 'not_supported')
        if security is None:
            # use config
            security = security_config.get(protocol_name, None)
            self.logger.info(f'!!Use default security: {security}')
        self.logger.info(f'Make FLEX {protocol_name} protocol using sec_param: {security}.')
        self._aligner = make_protocol(
            protocol_name,
            federal_info,
            sec_param=security
        )

    @ClassMethodAutoLog()
    def align(self, id_cols: List[List[str]], place_holder: str = '^_^') -> Union[None, List[List[str]]]:
        """
        Align sample in id_cols.

        Args:
            id_cols: list[list[str]], id columns (maybe one), ids are strings.
            place_holder: str, place holder to concatenate id cols.

        Returns:
            None or list[list[str]]: if this role is coord, return None, else return aligned ids.

        -----

        **Examples:**

        On Party1:
        >>> aligner = SampleAligner(federal_info, 2048)
        >>> id_cols = [['123', 'abc', '567', 'xxx']]
        >>> aligned = aligner.align(id_cols)
        >>> print(aligned)
        [['abc', '123']]

        On Party2:
        >>> aligner = SampleAligner(federal_info, 2048)
        >>> id_cols = [['yyy', 'abc', '123']]
        >>> aligned = aligner.align(id_cols)
        >>> print(aligned)
        [['abc', '123']]

        On Coordinator:
        >>> aligner = SampleAligner(federal_info, 2048)
        >>> aligner.align([[]])
        """
        # NOTE no check special char in ids
        # run align
        concated_id = self._concat_ids(id_cols, place_holder)
        concated_aligned = self._aligner.align(concated_id)
        return self._split_ids(concated_aligned, place_holder, len(id_cols) == 1)

    @ClassMethodAutoLog()
    def verify(self, id_cols: List[List[str]], place_holder: str = '^_^') -> Union[None, bool]:
        """
        Check alignment of samples in id_cols.

        Args:
            id_cols: list[list[str]], id columns (maybe one), ids are strings.
            place_holder: str, place holder to concatenate id cols.

        Returns:
            None or bool: if this role is coord, return None, else return True if samples are aligned.

        -----

        **Examples:**
        
        On Party1:
        >>> aligner = SampleAligner(federal_info, 2048)
        >>> id_cols = [['123', 'abc', '567', 'xxx']]
        >>> is_aligned = aligner.verfiy(id_cols)
        >>> print(is_aligned)
        False

        On Party2:
        >>> aligner = SampleAligner(federal_info, 2048)
        >>> id_cols = [['yyy', 'abc', '123']]
        >>> is_aligned = aligner.verify(id_cols)
        >>> print(is_aligned)
        False

        On Coordinator:
        >>> aligner = SampleAligner(federal_info, 2048)
        >>> aligner.verify([[]])
        """
        # NOTE no check special char in ids
        # run align
        concated_id = self._concat_ids(id_cols, place_holder)
        return self._aligner.verify(concated_id)

    @staticmethod
    def _concat_ids(id_cols: List[List[str]], place_holder: str) -> List[str]:
        # concat id_cols to concated_id
        if len(id_cols) == 1:
            concated_id = id_cols[0]
        else:
            concated_id = [place_holder.join((str(ii) for ii in i)) for i in zip(*id_cols)]
        return concated_id

    @staticmethod
    def _split_ids(concated_aligned: List[str], place_holder: str, single_col: bool = False) -> Union[
        None, List[List[str]]]:
        if concated_aligned is None:
            return None
        if single_col:
            return [concated_aligned]
        else:
            aligned = [i.split(place_holder) for i in concated_aligned]
            return [list(i) for i in zip(*aligned)]
